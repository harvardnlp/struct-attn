local CRF, parent = torch.class('nn.CRF', 'nn.Module')

function CRF:__init(nstates)
    parent.__init(self, nstates)

    self.nstates = nstates
    self.weight = torch.zeros(nstates, nstates)
    self.gradWeight = torch.zeros(nstates, nstates)
    self.gradInput = torch.Tensor()
    self.output = torch.Tensor()

    self._logGradInput = torch.Tensor()
    self._logGradInputSign = torch.Tensor()
    self._logGradInputTemp = torch.Tensor()
    self._logGradInputSignTemp = torch.Tensor()

    self._expandedWeight = torch.Tensor()
    self._gradExpandedWeight = torch.Tensor()
    self._logGradExpandedWeight = torch.Tensor()
    self._logGradExpandedWeightSign = torch.Tensor()
    self._logTables = torch.Tensor()
    self._logGradTables = torch.Tensor()
    self._logGradTablesSign = torch.Tensor()
    self._logForwardGrad = torch.Tensor()
    self._logBackwardGrad = torch.Tensor()
    self._logForwardGradSign = torch.Tensor()
    self._logBackwardGradSign = torch.Tensor()
    self._logMarginalProduct = torch.Tensor()

    self._nodeMarginals = torch.Tensor()
    self._nodeMarginalsSign = torch.Tensor()

    self._logbmm = torch.Tensor()
    self._logbmmSign = torch.Tensor()
    self._logB = torch.Tensor()
    self._logF = torch.Tensor()

    self._markov = Markov(nstates, 'torch.CudaTensor')

    self:reset()
end

function CRF:updateOutput(input)
    -- Input is the unary potentials of size batch x n x C
    local n, batches = input:size(2), input:size(1)

    if self:type() == 'torch.CudaTensor' and not package.loaded["cuda-mod"] then
        require "cuda-mod"
    end

    -- Expand the weights to form an entire lattice 
    self._expandedWeight:resize(batches, n+1, self.nstates, self.nstates)
    self._expandedWeight:copy(
        self.weight:view(-1, 1, self.nstates, self.nstates)
        :expandAs(self._expandedWeight))

    -- Handle start/end of lattice; weights are geometric mean of regular params
    self._expandedWeight[{{}, 1, 1}]:copy(torch.sum(self._expandedWeight[{{}, 1}],2)
        :div(self.nstates))
    self._expandedWeight[{{}, n+1, {}, 1}]:copy(torch.sum(self._expandedWeight[{{}, n+1}], 3)
        :div(self.nstates))
    self._expandedWeight[{{}, 1, {2, -1}}]:fill(-math.huge) 
    self._expandedWeight[{{}, n+1, {}, {2, -1}}]:fill(-math.huge)

    -- Run forward/backward
    self._markov:forwardBackward(input, self._expandedWeight, self._logTables)

    -- Combine tables into marginals
    self.output:resize(batches, n+1, self.nstates, self.nstates)
    self._markov:marginals(self.output, self._logTables, self._expandedWeight)

    Util.fixnan(self.output)
    return self.output
end

function CRF:updateGradInput(input, gradOutput)
    -- gradOutput is the gradient of the loss wrt marginals, size {batch x n+1 x C x C}
    local batches, n = gradOutput:size(1), gradOutput:size(2) - 1

    -- Switch to log space
    local logGradOutput = torch.abs(gradOutput):log()
    local logGradOutputSign = torch.sign(gradOutput)

    -- Account for the fact that we output log-marginals rather than marginals
    logGradOutput:add(-1, self.output)
    Util.fixnan(logGradOutput)

    -- Run dynamic programming to get intermediate gradient tables 
    self:updateGradTables(input, logGradOutput, logGradOutputSign)

    -- Initialize
    self._logGradInput:resize(batches, n, self.nstates):zero()
    self._logGradInputSign:resizeAs(self._logGradInput):fill(1)
    self._logGradInputTemp:resizeAs(self._logGradInput)
    self._logGradInputSignTemp:resizeAs(self._logGradInput)

    -- dL/d(node potential) = alpha*beta_hat + beta*alpha_hat (at that node)
    Util.logadd(self._logGradInput, self._logGradInputSign,
                    torch.add(self._logTables[{1, {2, -2}}]:transpose(1, 2),
                                 self._logBackwardGrad[{{}, {2, -1}}]),
                    torch.add(self._logTables[{2, {2, -2}}]:transpose(1, 2),
                                 self._logForwardGrad[{{}, {1, -2}}]),
                    self._logBackwardGradSign[{{}, {2, -1}}],
                    self._logForwardGradSign[{{}, {1, -2}}],
                    self._logGradInputTemp, self._logGradInputSignTemp)

    -- Account for dL/dZ
    local factor = Util.logsumNumber(self._logMarginalProduct, logGradOutputSign)
    factor:add(-1, self._logTables[{1, n+2, {}, 1}])

    local nodes = self._logTables[{1,  {2, -2}}]:transpose(1, 2)
    self._nodeMarginals:resizeAs(nodes):copy(nodes)
        :add(self._logTables[{2, {2, -2}}]:transpose(1, 2))

    self._nodeMarginals:add(-1, input)
    self._nodeMarginals:add(factor:view(-1, 1, 1):expandAs(self._nodeMarginals))
    self._nodeMarginalsSign:typeAs(self._nodeMarginals):resizeAs(self._nodeMarginals):fill(-1)

    Util.logadd(self._logGradInput, self._logGradInputSign,
                    self._logGradInput, self._nodeMarginals,
                    self._logGradInputSign, self._nodeMarginalsSign,
                    self._logGradInputTemp, self._logGradInputSignTemp)

    self.gradInput:resize(batches, n, self.nstates)
    self.gradInput:copy(self._logGradInput):exp():cmul(self._logGradInputSign)
    Util.fixnan(self.gradInput)
    return self.gradInput
end

-- Set the gradWeight
function CRF:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    local batches, n, C = input:size(1), input:size(2), self.nstates

    local logGradOutputSign = torch.sign(gradOutput)

    -- dL/d(edge potential) = alpha*backwardGrad*w + beta*forwardGrad*w + dL/dM*dM/dw 
    self._logGradExpandedWeight:resize(n+1, batches, C, C):fill(-math.huge)
    self._logGradExpandedWeightSign:resize(n+1, batches, C, C):fill(1)

    -- these tensors are used to batch the two computations done in this loop
    self._logbmm:resize(batches, C, C):typeAs(input)
    self._logbmmSign:resize(batches, C, C):fill(1)

    self._logB = self._logB:resizeAs(input):copy(input):add(self._logBackwardGrad[{{}, {2, -1}}])
          :view(batches, -1, 1, C):transpose(1,2):contiguous()
    self._logF = self._logF:resizeAs(input):copy(input):add(self._logForwardGrad[{{}, {1, -2}}])
          :view(batches, -1, C, 1):transpose(1,2):contiguous()

    local forward = self._logTables:view(2, -1, batches,  C, 1)[1]
    local backward = self._logTables:view(2, -1, batches,  1, C)[2]

    -- temp variables
    local ones1 = torch.ones(self._logTables[{1,  1}]:size()):typeAs(self._logTables)
    local ones2 = torch.ones(self._logTables[{2,  1}]:size()):typeAs(self._logForwardGradSign)
    local t1a = torch.Tensor(batches, forward[1]:size(2), forward[1]:size(3), self._logB[1]:size(3)):typeAs(self._logTables)
    local t1b = torch.Tensor(batches, forward[1]:size(2), forward[1]:size(3), self._logB[1]:size(3)):typeAs(self._logTables)
    local t1 = self._logbmm:clone()
    local t2 = self._logbmm:clone()

    for i = 1, n do
        Util.logbmm(self._logbmm, self._logbmmSign, forward[i], self._logB[i],
                        ones1, self._logBackwardGradSign:select(2, i+1),
                        t1a, t1b)
        Util.logadd(self._logGradExpandedWeight[i],
                        self._logGradExpandedWeightSign[i],
                        self._logGradExpandedWeight[i],
                        self._logbmm, self._logGradExpandedWeightSign[i], self._logbmmSign,
                        t1, t2)

        Util.logbmm(self._logbmm, self._logbmmSign, self._logF[i], backward[i+2],
                        self._logForwardGradSign:select(2, i), ones2,
                        t1a, t1b)
        Util.logadd(self._logGradExpandedWeight[i+1],
                        self._logGradExpandedWeightSign[i+1],
                        self._logGradExpandedWeight[i+1],
                        self._logbmm, self._logGradExpandedWeightSign[i+1], self._logbmmSign,
                        t1, t2)
    end
    
    -- multiply by w and add dL/d(marginal)*d(marginal)/d(edge)
    t1 = self._logGradExpandedWeight:clone()
    self._logGradExpandedWeight:add(self._expandedWeight:transpose(1,2))

    Util.logadd(self._logGradExpandedWeight, self._logGradExpandedWeightSign,
                    self._logGradExpandedWeight, self._logMarginalProduct:transpose(1,2),
                    self._logGradExpandedWeightSign, logGradOutputSign:transpose(1,2),
                    nil, nil)

    -- partial with respect to the partition
    local factor = Util.logsumNumber(self._logMarginalProduct, logGradOutputSign)

    Util.logadd(self._logGradExpandedWeight, self._logGradExpandedWeightSign,
                    self._logGradExpandedWeight,
                    torch.add(self.output, factor:view(-1, 1, 1, 1)
                        :expandAs(self.output)):transpose(1, 2),
                    self._logGradExpandedWeightSign,
                    torch.Tensor(gradOutput:size()):typeAs(gradOutput):fill(-1))

    -- come out of log space
    self._gradExpandedWeight:resize(batches, n+1, C, C)
    self._gradExpandedWeight:copy(self._logGradExpandedWeight:exp()
        :cmul(self._logGradExpandedWeightSign):transpose(1, 2))

    -- handle edge cases at start node
    -- initial and final weights are combinations of other terms, so we expand to capture this
    self._gradExpandedWeight[{{}, 1, 1}]:div(self.nstates)
    self._gradExpandedWeight[{{}, n+1, {}, 1}]:div(self.nstates)
    self._gradExpandedWeight[{{}, 1, {2, -1}}]:copy(
        self._gradExpandedWeight[{{}, 1, 1}]:contiguous():view(-1, 1, C)
        :expand(batches, C-1, C))
    self._gradExpandedWeight[{{}, n+1, {}, {2, -1}}]:copy(
        self._gradExpandedWeight[{{}, n+1, {}, 1}]:contiguous():view(-1, C, 1)
            :expand(batches, C, C-1))

    self.gradWeight:add(scale, self._gradExpandedWeight:sum(1):sum(2):squeeze())
    Util.fixnan(self.gradWeight)
end

function CRF:updateGradTables(input, logGradOutput, logGradOutputSign)
    -- calculate dL/dalpha, dL/dbeta, backwardGrad, forwardGrad
    -- note: self._gradTables has size n+1 rather than n+2; the final node is ignored for forward
    -- and the initial node is ignored for backward because they don't affect the marginals.
    -- the initial for forward and final for backward are kept for convenience in
    -- backwardGrad and forwardGrad.

    local batches, n = input:size(1), input:size(2)

    -- marginalProduct = marginal * dL/d(marginal)
    self._logMarginalProduct:resizeAs(self.output):zero()
    self._logMarginalProduct:add(logGradOutput, self.output)
    Util.fixnan(self._logMarginalProduct)

    -- compute gradient of loss with respect to forward and backward
    -- 1. accumulate marginalProduct terms with the same forward term
    -- 2. divide by the relevant forward terms
    -- 3. accumulate marginalProduct terms with the same backward term
    -- 4. divide by the relevant backward terms
    -- 5. zero out ends
    self._logGradTables:resize(2, n+1, batches, self.nstates):zero()
    self._logGradTablesSign:resizeAs(self._logGradTables):fill(0)

    local logsum, logsumSign = Util.logsum(self._logMarginalProduct, 4, logGradOutputSign)
    self._logGradTables[1]:add(logsum:view(batches, n+1, self.nstates):transpose(1,2))
    self._logGradTablesSign[1]:add(logsumSign:view(batches, n+1, self.nstates):transpose(1,2))
    self._logGradTables[1]:add(-1, self._logTables[{1,  {1, -2}}])

    logsum, logsumSign = Util.logsum(self._logMarginalProduct, 3, logGradOutputSign)
    self._logGradTables[2]:add(logsum:view(batches, n+1, self.nstates):transpose(1,2))
    self._logGradTablesSign[2]:add(logsumSign:view(batches, n+1, self.nstates):transpose(1,2))
    self._logGradTables[2]:add(-1, self._logTables[{2,  {2, -1}}])
    self._logGradTables[{1, 1, {}, {2, -1}}]:fill(-math.huge)
    self._logGradTables[{2, n+1, {}, {2, -1}}]:fill(-math.huge)

    -- Run dynamic programming
    if self._logForwardGrad:dim() > 0 then
        self._logForwardGrad = self._logForwardGrad:transpose(1,2)
        self._logForwardGradSign = self._logForwardGradSign:transpose(1,2)
        self._logBackwardGrad = self._logBackwardGrad:transpose(1,2)
        self._logBackwardGradSign = self._logBackwardGradSign:transpose(1,2)
    end

    self._markov:makePotentials(input, self._expandedWeight, true)
    self._markov:backwardGrad(self._logBackwardGrad, self._logBackwardGradSign,
                                      self._logGradTables[1], self._logGradTablesSign[1])
    self._markov:forwardGrad(self._logForwardGrad, self._logForwardGradSign,
                                     self._logGradTables[2], self._logGradTablesSign[2])

    self._logForwardGrad = self._logForwardGrad:transpose(1,2)
    self._logForwardGradSign = self._logForwardGradSign:transpose(1,2)
    self._logBackwardGrad = self._logBackwardGrad:transpose(1,2)
    self._logBackwardGradSign = self._logBackwardGradSign:transpose(1,2)
end
