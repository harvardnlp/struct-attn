local Markov, parent = torch.class("Markov", "nn.Module")
-- Fake being a module to inherit :type()

function Markov:__init(nstates, typ)
    self.nstates = nstates
    self._tempDp = torch.Tensor()
    self._tempDpGrad1 = torch.Tensor()
    self._tempDpGrad2 = torch.Tensor()
    self._tempMarginals = torch.Tensor()
    self._ones = torch.Tensor()

    self._forwardPotentials = torch.Tensor()
    self._backwardPotentials = torch.Tensor()

end

-- Run Viterbi algorithm
function Markov.viterbi(logMarginals)
    local useCuda = (logMarginals:type() == 'torch.CudaTensor')

    local batches = logMarginals:size(1)
    local n = logMarginals:size(2) - 1
    local C = logMarginals:size(3)

    -- convert to conditional marginals
    local logConditionalMarginals = logMarginals:clone()
    margSum, _ = Util.logsum(logConditionalMarginals, 4)
    logConditionalMarginals:add(-1, margSum:expandAs(logConditionalMarginals))

    local bestPaths = torch.Tensor(batches, n, C):typeAs(logMarginals)
    local backPointers
    if useCuda then
        backPointers = torch.CudaTensor(batches, n, C)
    else
        backPointers = torch.LongTensor(batches, n, C)
    end

    -- compute dynamic programming tables
    bestPaths[{{}, 1}]:copy(logConditionalMarginals[{{}, 1, 1}])
    backPointers[{{}, 1}]:fill(1)
    for i = 2, n do
        m, idx = torch.max(torch.add(bestPaths[{{}, i-1}]:contiguous():view(-1, C, 1)
            :expand(batches, C, C), logConditionalMarginals[{{}, i}]), 2)
        bestPaths[{{}, i}]:copy(m)
        backPointers[{{}, i}]:copy(idx)
    end

    -- compute paths from backpointers
    local paths
    paths = torch.LongTensor(batches, n)
    bestScore, paths[{{}, n}] = bestPaths[{{}, n}]:double():max(2)

    for i = n-1, 1, -1 do
        for b = 1, batches do
            paths[b][i] = backPointers[b][i+1][paths[b][i+1]]
        end
    end
    if useCuda then
        return torch.exp(bestScore), paths:cuda()
    end
    return torch.exp(bestScore), paths    
end

-- Sample from the marginal distribution
function Markov.sample(logMarginals)
    local batches = logMarginals:size(1)
    local n = logMarginals:size(2) - 1
    local C = logMarginals:size(3)

    conditionalMarginals = torch.exp(logMarginals)
    conditionalMarginals:cdiv(conditionalMarginals:sum(4):expandAs(conditionalMarginals))

    local sample = torch.Tensor(batches, n)
    local prob = 1
    local lastTag = 1
    for batch = 1, batches do
        for i = 1, n do
            sample[batch][i] = torch.multinomial(conditionalMarginals[batch][i][lastTag], 1):squeeze()
            prob = prob*conditionalMarginals[batch][i][lastTag][sample[batch][i]]
            lastTag = sample[batch][i]
        end
    end

    return prob, sample
end

-- Build potentials (node and edge)
function Markov:makePotentials(unaryPotentials, binaryPotentials, grad)
    local batches, n, C = get_sizes_uni(unaryPotentials)
    local binaryPotentials = binaryPotentials:transpose(1, 2)

    self._backwardPotentials:resizeAs(binaryPotentials)
    self._forwardPotentials:resizeAs(binaryPotentials)
    local unaryPotentials = unaryPotentials:view(-1, n, 1, C):transpose(1, 2):expand(n, batches,  C, C)

    self._backwardPotentials:copy(binaryPotentials)
    if not grad then
        self._backwardPotentials[{{2, -1}}]:add(unaryPotentials:transpose(3,4))
    else
        self._backwardPotentials[{{1, -2}}]:transpose(3, 4):add(unaryPotentials:transpose(3,4))
    end
    self._backwardPotentials[{1, {}, {2, -1}}]:fill(-math.huge)
    self._backwardPotentials[{n+1, {}, {}, {2, -1}}]:fill(-math.huge)

    -- Forward
    self._forwardPotentials:transpose(3, 4):copy(binaryPotentials)
    if not grad then
        self._forwardPotentials[{{1, -2}}]:transpose(3, 4):add(unaryPotentials)
    else
        self._forwardPotentials[{{2, -1}}]:add(unaryPotentials)
    end
    self._forwardPotentials[{1, {}, {}, {2, -1}}]:fill(-math.huge)
    self._forwardPotentials[{n+1, {}, {2, -1}}]:fill(-math.huge)
end

-- Run forward algorithm
function Markov:forward(forwardTable)
    local batches, n, C = get_sizes_dp(forwardTable)
    potentials = self._forwardPotentials

    -- compute forward table with dynamic programming
    -- initialize with 1 0 ... 0 (in log space equivalent)

    -- local forwardTable
    forwardTable = forwardTable:view(n+2, batches, C, 1)

    forwardTable:fill(-math.huge)
    forwardTable[{1, {}, 1}]:zero()


    self._tempDp:resize(batches, potentials[1]:size(2),
                              forwardTable[1]:size(3), potentials[1]:size(3))
    for i = 2, n+2 do
        Util.logbmm2(forwardTable[i],
                         potentials[i-1],
                         forwardTable[i-1],
                         self._tempDp)
    end
    forwardTable[{n+2, {},  {2, -1}}]:fill(-math.huge)
end

-- Run backward algorithm
function Markov:backward(backwardTable)
    local batches, n, C = get_sizes_dp(backwardTable)
    potentials = self._backwardPotentials

    -- compute backward table with dynamic programming
    -- initialize with 1 0 ... 0
    backwardTable = backwardTable:view(n+2, batches, C, 1)
    backwardTable:fill(-math.huge)
    backwardTable[{n+2, {}, 1}]:zero()

    self._tempDp:resize(batches, potentials[1]:size(2),
                              backwardTable[1]:size(3), potentials[1]:size(3))
    for i = n+1, 1, -1 do
        Util.logbmm2(backwardTable[i],
                         potentials[i],
                         backwardTable[i+1], self._tempDp)
    end
    backwardTable[{1, {}, {2, -1}}]:fill(-math.huge)
end

-- Run forward/backward
function Markov:forwardBackward(unaryPotentials, binaryPotentials, logTables)
    -- run forward and backward
    local batches, n, C = get_sizes_uni(unaryPotentials)
    logTables:resize(2, n+2, batches, self.nstates)
    self:makePotentials(unaryPotentials, binaryPotentials)
    self:forward(logTables[1])
    self:backward(logTables[2])
end

-- Compute marginals
function Markov:marginals(marginals, logTables, binaryPotentials)
    -- combine tables into marginals
    local batches, n, C = get_sizes_dp(logTables[1])

    -- multiply forward, backward, and edge weights
    local forward = logTables:view(2, -1, batches,  C, 1)[1]
    local backward = logTables:view(2, -1, batches, 1, C)[2]
    marginals:zero()

    self._tempMarginals:resize(batches, forward[1]:size(2),
                                        backward[1]:size(3), forward[1]:size(3))
    for i = 1, n+1 do
        Util.logbmm2(marginals:select(2, i),
                         forward[i],
                         backward[i+1], self._tempMarginals)
    end
    marginals:add(binaryPotentials)

    -- divide by partition to normalize
    marginals:add(-1, logTables[{1, n+2, {}, 1}]:contiguous():view(-1,1,1,1)
        :expandAs(marginals))
    marginals[{{}, 1, {2, -1}}]:fill(-math.huge)
    marginals[{{}, n+1, {}, {2, -1}}]:fill(-math.huge)
end

-- Run backward dynamic programming (intermediate in gradient computation)
function Markov:backwardGrad(logBackwardGrad, logBackwardGradSign,
                                      logGradForward, logGradForwardSign)
    -- gradForward is the gradient of the loss wrt to the forward table
    local batches, n, C = get_sizes_dp_grad(logGradForward)
    local potentials = self._backwardPotentials

    -- initialize dynamic programming
    logBackwardGrad:resize(n+1, batches,  C, 1)
    logBackwardGradSign:resize(n+1, batches,  C, 1)
    logBackwardGrad[n+1]:copy(logGradForward[n+1])
    logBackwardGradSign[n+1]:copy(logGradForwardSign[n+1])


    -- Make temp variables
    self._ones:resize(potentials[1]:size()):contiguous():fill(1)
    self._tempDpGrad1:resize(batches, potentials[1]:size(2),
                                     logBackwardGrad[1]:size(2)+1, logBackwardGrad[1]:size(3)):contiguous()
    self._tempDpGrad2:resizeAs(self._tempDpGrad1):contiguous()

    -- recursive step
    for i = n, 1, -1 do
        self._tempDpGrad1:narrow(3, 1, 1):squeeze():copy(logGradForward[i])
        self._tempDpGrad2:narrow(3, 1, 1):squeeze():copy(logGradForwardSign[i])
        Util.logbmm(logBackwardGrad[i], logBackwardGradSign[i],
                        potentials[i], logBackwardGrad[i+1],
                        self._ones, logBackwardGradSign[i+1],
                        self._tempDpGrad1, self._tempDpGrad2, 2, 2)
    end
end

-- Run forward dynamic programming (intermediate in gradient computation)
function Markov:forwardGrad(logForwardGrad, logForwardGradSign,
                                     logGradBackward, logGradBackwardSign)
    -- gradBackward is the gradient of the loss wrt to the backward table
    local batches, n, C = get_sizes_dp_grad(logGradBackward)
    local potentials = self._forwardPotentials

    -- initialize dynamic programming
    logForwardGrad:resize(n+1, batches,  C, 1)
    logForwardGradSign:resize(n+1, batches, C, 1)
    logForwardGrad[1]:copy(logGradBackward[1])
    logForwardGradSign[1]:copy(logGradBackwardSign[1])

    -- Make temp variables
    self._ones:resize(potentials[1]:size()):fill(1)
    self._tempDpGrad1:resize(batches, potentials[1]:size(2),
                                    logForwardGrad[1]:size(2) + 1, logForwardGrad[1]:size(3))
    self._tempDpGrad2:resizeAs(self._tempDpGrad1)

    -- recursive step
    for i = 2, n+1 do
        self._tempDpGrad1:narrow(3, 1, 1):squeeze():copy(logGradBackward[i])
        self._tempDpGrad2:narrow(3, 1, 1):squeeze():copy(logGradBackwardSign[i])
        Util.logbmm(logForwardGrad[i], logForwardGradSign[i],
                        potentials[i],
                        logForwardGrad[i-1],
                        self._ones,
                        logForwardGradSign[i-1],
                        self._tempDpGrad1, self._tempDpGrad2, 2, 2)
    end
end

-- Convenience functions for sizing
function get_sizes_uni(unaryPotentials)
    local batches = unaryPotentials:size(1)
    local n = unaryPotentials:size(2)
    local C = unaryPotentials:size(3)
    return batches, n, C
end

function get_sizes_dp(forwardTable)
    local batches = forwardTable:size(2)
    local n = forwardTable:size(1) - 2
    local C = forwardTable:size(3)
    return batches, n, C
end

function get_sizes_dp_grad(forwardTable)
    local batches = forwardTable:size(2)
    local n = forwardTable:size(1) - 1
    local C = forwardTable:size(3)
    return batches, n, C
end


