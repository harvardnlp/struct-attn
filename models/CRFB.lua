-- A version of CRF which accepts binary potentials as input and has no weights
local CRFB, parent = torch.class('nn.CRFB', 'nn.Module')

function CRFB:__init()
    parent.__init(self)

    self.gradInput = torch.Tensor()
    self.output = torch.Tensor()

    self._unaryPotentials = torch.Tensor()
    self._logGradInput = torch.Tensor()
    self._logGradInputSign = torch.Tensor()
    self._logGradInputTemp = torch.Tensor()
    self._logGradInputSignTemp = torch.Tensor()

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

function CRFB:updateOutput(input)
    -- Input is the binary potentials of size batch x (n + 1) x C x C 
    local batches = input:size(1)
    local n = input:size(2) - 1
    local nstates = input:size(3)
    assert(nstates == input:size(4), "invalid input, the input size must be [batch x (n + 1) x C x C]")
    self._unaryPotentials:resize(batches, n, nstates):zero()

    if self:type() == 'torch.CudaTensor' and not package.loaded["cuda-mod"] then
        require "cuda-mod"
    end

    -- Run forward/backward
    self._markov:forwardBackward(self._unaryPotentials, input, self._logTables)

    -- Combine tables into marginals
    self.output:resize(batches, n+1, nstates, nstates)
    self._markov:marginals(self.output, self._logTables, input)

    Util.fixnan(self.output)
    return self.output
end

function CRFB:updateGradInput(input, gradOutput)
    -- gradOutput is the gradient of the loss wrt marginals, size {batch x n+1 x C x C}
    local batches, n, C = gradOutput:size(1), gradOutput:size(2) - 1, gradOutput:size(3)

    -- Switch to log space
    local logGradOutput = torch.abs(gradOutput):log()
    local logGradOutputSign = torch.sign(gradOutput)

    -- Account for the fact that we output log-marginals rather than marginals
    logGradOutput:add(-1, self.output)
    Util.fixnan(logGradOutput)

    -- Run dynamic programming to get intermediate gradient tables 
    self:updateGradTables(input, logGradOutput, logGradOutputSign)

    -- Set the gradWeight

    -- dL/d(edge potential) = alpha*backwardGrad*w + beta*forwardGrad*w + dL/dM*dM/dw 
    self._logGradExpandedWeight:resize(n+1, batches, C, C):fill(-math.huge)
    self._logGradExpandedWeightSign:resize(n+1, batches, C, C):fill(1)

    -- these tensors are used to batch the two computations done in this loop
    self._logbmm:resize(batches, C, C):typeAs(input)
    self._logbmmSign:resize(batches, C, C):fill(1)

    self._logB = self._logB:resizeAs(self._unaryPotentials):copy(self._unaryPotentials):add(self._logBackwardGrad[{{}, {2, -1}}])
          :view(batches, -1, 1, C):transpose(1,2):contiguous()
    self._logF = self._logF:resizeAs(self._unaryPotentials):copy(self._unaryPotentials):add(self._logForwardGrad[{{}, {1, -2}}])
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
    self._logGradExpandedWeight:add(input:transpose(1,2))

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
    self.gradInput:resize(batches, n+1, C, C)
    self.gradInput:copy(self._logGradExpandedWeight:exp()
        :cmul(self._logGradExpandedWeightSign):transpose(1, 2))
    Util.fixnan(self.gradInput)
    return self.gradInput
end

function CRFB:updateGradTables(input, logGradOutput, logGradOutputSign)
    -- calculate dL/dalpha, dL/dbeta, backwardGrad, forwardGrad
    -- note: self._gradTables has size n+1 rather than n+2; the final node is ignored for forward
    -- and the initial node is ignored for backward because they don't affect the marginals.
    -- the initial for forward and final for backward are kept for convenience in
    -- backwardGrad and forwardGrad.

    local batches, n, nstates = input:size(1), input:size(2) - 1, input:size(3)

    assert(
        self._unaryPotentials:size(1) == batches and
        self._unaryPotentials:size(2) == n and
        self._unaryPotentials:size(3) == nstates,
        "Unary potentials not initialized properly, size mismatch"
    )

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
    self._logGradTables:resize(2, n+1, batches, nstates):zero()
    self._logGradTablesSign:resizeAs(self._logGradTables):fill(0)

    local logsum, logsumSign = Util.logsum(self._logMarginalProduct, 4, logGradOutputSign)
    self._logGradTables[1]:add(logsum:view(batches, n+1, nstates):transpose(1,2))
    self._logGradTablesSign[1]:add(logsumSign:view(batches, n+1, nstates):transpose(1,2))
    self._logGradTables[1]:add(-1, self._logTables[{1,  {1, -2}}])

    logsum, logsumSign = Util.logsum(self._logMarginalProduct, 3, logGradOutputSign)
    self._logGradTables[2]:add(logsum:view(batches, n+1, nstates):transpose(1,2))
    self._logGradTablesSign[2]:add(logsumSign:view(batches, n+1, nstates):transpose(1,2))
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

    self._markov:makePotentials(self._unaryPotentials, input, true)
    self._markov:backwardGrad(self._logBackwardGrad, self._logBackwardGradSign,
                                      self._logGradTables[1], self._logGradTablesSign[1])
    self._markov:forwardGrad(self._logForwardGrad, self._logForwardGradSign,
                                     self._logGradTables[2], self._logGradTablesSign[2])

    self._logForwardGrad = self._logForwardGrad:transpose(1,2)
    self._logForwardGradSign = self._logForwardGradSign:transpose(1,2)
    self._logBackwardGrad = self._logBackwardGrad:transpose(1,2)
    self._logBackwardGradSign = self._logBackwardGradSign:transpose(1,2)
end
