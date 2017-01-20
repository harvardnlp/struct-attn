require("hdf5")
require("nn")
require("optim")
require("rnn")
require("nngraph")
require 'models/Util.lua'
require 'models/Markov.lua'
require "models/CRFB.lua"

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'qa11.hdf5', 'data file')
cmd:option('-classifier', 'binarycrf', 'classifier to use')

-- Hyperparameters
cmd:option('-eta',0.01,'learning rate hyperparameter for lr/nn')
cmd:option('-max_grad_norm',40, 'max norm for RNN models')
cmd:option('-grad_norm','global', '[global, local, off] gradient renormalization')
cmd:option('-N',100,'num epochs hyperparameter for lr/nn')
cmd:option('-D0',20,'num outputs of lookup layer of nn')
cmd:option('-debug','','specify model to debug')
cmd:option('-unit_test',false,'whether run all unit tests')

-- Model parameters
cmd:option('-max_history',25,'max history')
cmd:option('-max_sent_len',20,'max sent len')
cmd:option('-pe',false,'enable position encoding')
cmd:option('-te',false,'enable temporal encoding')
cmd:option('-save',false,'whether to save model')
cmd:option('-saveminacc',0.5,'minimum accuracy on test set required to save model')
cmd:option('-cuda',false,'whether to use cuda')

function main()
  -- Parse input params
  opt = cmd:parse(arg)
  load()
  if opt.debug ~= '' then
    debugModel()
  else
    print(string.format("Using data file: %s", opt.datafile))
    if opt.cuda then
      require("cunn")
      print("Using Cuda")
    else
      print("Using CPU")
    end
    if opt.pe then
      print('Using Position Encoding')
    end
    if opt.te then
      print('Using Temporal Encoding')
    end
    if opt.classifier == 'binarycrf' then
      runCRF()
    elseif opt.classifier == 'unarycrf' then
      runCRF()
    else
      print("NOT IMPLEMENTED")
    end
  end
end

function runCRF()
  trainCRF()
  testModel(false)
end

function trainCRF()
  print(string.format(
    "Training %s model with N = %d, D0 = %d, Eta = %f, "..
    "Max History = %d, Max Sentence Length = %d, Max Gradient Norm = %d, "..
    "Gradient Norm Type = %s", opt.classifier,
    opt.N, opt.D0, opt.eta, opt.max_history, opt.max_sent_len,
    opt.max_grad_norm, opt.grad_norm))

  D0 = opt.D0
  local eta = opt.eta
  local trainLoss = torch.zeros(opt.N)
  local timer = torch.Timer()

  create_embedding_layers()

  -- number of states in CRF, including the <nil> state at start and end time step
  nstates = opt.max_history + 1
  -- number of time steps, i.e. length of sequence
  T = 2

  if opt.classifier == 'binarycrf' then
    create_binarycrf_model()
  else
    create_unarycrf_model()
  end

  if opt.classifier == 'binarycrf' then
    theta0 = torch.zeros(1, nstates, nstates)
    theta12 = torch.zeros(1, nstates, nstates)
    theta3 = torch.zeros(1, nstates, nstates)
  end

  if opt.cuda then
    model:cuda() 
    crit = crit:cuda()
    if opt.classifier == 'binarycrf' then
      theta0 = theta0:cuda()
      theta12 = theta12:cuda()
      theta3 = theta3:cuda()
    end
    train_stories = train_stories:cuda()
    train_questions = train_questions:cuda()
    train_answers = train_answers:cuda()
  end

  params, gradParams = model:getParameters()

  -- good initializations for params
  gradParams:zero()
  for i=1, params:size(1) do params[i] = torch.randn(1)[1]/ torch.sqrt(10) end -- IMPORTANT

  if opt.pe then
    resetPE()
  end

  for iEpoch = 1, opt.N do
    local idxs = torch.randperm(train_stories:size(1)):long()
    for l = 1, train_stories:size(1) do
      zeroLookupTable()

      i = idxs[l]

      local x = getStory(train_stories, i, opt.max_history, opt.max_sent_len)
      local q = train_questions[i]
      local preds = model:forward({x,q,te_mask,theta0,theta12,theta3})

      -- test various outputs
      if opt.unit_test then
        for i,node in ipairs(model.forwardnodes) do
          if node.data.annotations.name == 'cp123' then
            assert(torch.abs(node.data.module.output:sum()-1) <= 1e-8,
              'incorrect distribution')
            break
          end
          if node.data.annotations.name == 'nom' then
            local nom_output = node.data.module.output:view(T,nstates-1)
            assert(torch.all(nom_output:sum(2):add(-1):abs():le(1e-8)),
              'incorrect node marginals')
            break
          end
        end

        -- embedding rows corresponding to padding should be zero
        nng_x = get_module_output('x')
        nng_m = get_module_output('m')
        nng_c = get_module_output('c')
        nng_c2 = get_module_output('c2')
        nng_c3 = get_module_output('c3')
        n_pad_rows = torch.sum(nng_x[{{},1}]:eq(idx_pad))
        if nng_m then assert(n_pad_rows == n_zero_rows(nng_m), 'Embeddings incorrect.') end
        if nng_c then assert(n_pad_rows == n_zero_rows(nng_c), 'Embeddings incorrect.') end
        if nng_c2 then assert(n_pad_rows == n_zero_rows(nng_c2), 'Embeddings incorrect.') end
        if nng_c3 then assert(n_pad_rows == n_zero_rows(nng_c3), 'Embeddings incorrect.') end

        nng_theta = get_module_output('theta') -- unary crf model
        if nng_theta then
          assert(T*(n_pad_rows+1) == nng_theta[{1,{},{1,n_pad_rows+1}}]:eq(-math.huge):sum(), 'incorrect unary potentials')
        end
      end

      if l < 2 then
        local debug_names = Set { "x", "m", "theta", "cp123", "CRF", "Theta01", "Theta12", "Theta20" }
        for i,v in ipairs(model.forwardnodes) do
          if debug_names[v.data.annotations.name] then
            print(v.data.annotations.name)
            print(v.data.module.output)
          end
        end
      end
      
      local max_prob, max_idx = torch.max(preds,1)
      local loss = crit:forward(preds,train_answers[i])
      trainLoss[iEpoch] = trainLoss[iEpoch] + loss
      local dLdinp = crit:backward(preds,train_answers[i])
      gradParams:zero()
      model:backward({x,q,te_mask,theta0,theta12,theta3},dLdinp)
      zeroPEGrad()
      adjustGrad()
      params:add(-eta, gradParams)
    end

    print("epoch "..iEpoch,"loss "..trainLoss[iEpoch])
    print("Total time taken",timer:time().real)

    if trainLoss[iEpoch] < 5 then
      print("Loss is sufficiently small. Exiting")
      break
    end
    
    if iEpoch == 20 then
      if opt.classifier == 'binarycrf' then
        print('Re-inserting softmax')
        THETA01_pre:insert(nn.LogSoftMax(), 3)
        THETA12_pre:insert(nn.LogSoftMax(), 2)
        THETA20_pre:insert(nn.LogSoftMax(), 3)
        print('THETA01_pre')
        print(THETA01_pre)
        print('THETA12_pre')
        print(THETA12_pre)
        print('THETA20_pre')
        print(THETA20_pre)
      else
        THETA_PRE:insert(nn.LogSoftMax(), 3)
      end
    end

    testModel(true)

    eta = adjustEta(eta, iEpoch)
  end
end

function create_binarycrf_model()
  -- theta^(t) (i,j) = u_1^T m_i + m_i^T n_j + u_2^T n_j 
  M_I = nn.Sequential():add(nn.SplitTable(1)):add(nn.MapTable()
    :add(nn.Replicate(opt.max_history, 1))):add(nn.JoinTable(1))

  N_J = nn.Sequential():add(nn.Replicate(opt.max_history, 1))
    :add(nn.Reshape(opt.max_history * opt.max_history, D0))

  U2 = nn.Linear(D0,D0)

  U_MI = nn.MM(false,true) -- Mi * u
  MI_NJ = nn.DotProduct() -- Mi * Mj
  U_NJ = nn.MM(false, true) -- Nj * u

  THETA01_pre = nn.Sequential()
    :add(nn.MM(false,true))
    :add(nn.View(1,opt.max_history))
    :add(nn.Padding(2,-1,2,-math.huge))
    :add(nn.Padding(1,opt.max_history,2,-math.huge))
    :add(nn.View(1,nstates,nstates))
  THETA01 = nn.CAddTable()

  THETA12_pre = nn.Sequential()
    :add(nn.CAddTable())
    :add(nn.View(opt.max_history,opt.max_history))
    :add(nn.Padding(1,-1,2,-math.huge))
    :add(nn.Padding(2,-1,2,-math.huge))
    :add(nn.View(1,nstates,nstates))
  THETA12 = nn.CAddTable()

  THETA20_pre = nn.Sequential()
    :add(nn.MM(false,true))
    :add(nn.View(1,opt.max_history))
    :add(nn.View(opt.max_history,1))
    :add(nn.Padding(1,-1,2,-math.huge))
    :add(nn.Padding(2,opt.max_history,2,-math.huge))
    :add(nn.View(1,nstates,nstates))
  THETA20 = nn.CAddTable()

  CRF = nn.Sequential():add(nn.JoinTable(1)):add(nn.View(1,T+1,nstates,nstates))
    :add(nn.NaN(nn.CRFB())):add(nn.Exp())

  create_crf_infer_module()

  -- nngraph based model
  create_embedding_nodes()
  T0_inp = nn.Identity()():annotate({name = 'T0', description  = 'constant binary potentials for first step'})
  T12_inp = nn.Identity()():annotate({name = 'T12', description  = 'constant binary potentials for second step'})
  T3_inp = nn.Identity()():annotate({name = 'T3', description  = 'constant binary potentials for last step'})

  m_i = M_I(m):annotate({name = 'mi', description = 'memory embeddings replicated'})
  n_j = N_J(c):annotate({name = 'nj', description = 'memory embeddings replicated'})

  u2 = U2(u):annotate({name = 'u2', description = 'second layer query embeddings'})

  umi = U_MI({m_i,u}):annotate({name = 'umi', description = 'u^T m_i'})
  minj = MI_NJ({m_i,n_j}):annotate({name = 'minj', description = 'm_i^T n_j'})
  unj = U_NJ({n_j,u2}):annotate({name = 'unj', description = 'u^T n_j'})
  
  theta01_pre = THETA01_pre({u,m}):annotate({name = 'Theta01_pre', description = 'binary potential theta at first step'})
  theta01 = THETA01({theta01_pre,T0_inp}):annotate({name = 'Theta01', description = 'binary potential theta at first step'})

  theta12_pre = THETA12_pre({umi,minj,unj}):annotate({name = 'Theta12_pre', description = 'binary potential theta'})
  theta12 = THETA12({theta12_pre, T12_inp}):annotate({name = 'Theta12', description = 'binary potential theta'})

  theta20_pre = THETA20_pre({u2,c}):annotate({name = 'Theta20_pre', description = 'binary potential theta at last step'})
  theta20 = THETA20({T3_inp,theta20_pre}):annotate({name = 'Theta20', description = 'binary potential theta at last step'})
  
  crf = CRF({theta01,theta12,theta20}):annotate({name = 'CRF', description = 'CRF layer'})

  create_crf_infer_node()

  model = nn.gModule({x_inp,q_inp,te_inp,T0_inp,T12_inp,T3_inp},{a_hat})
end

function create_unarycrf_model()
  U_MI_1 = nn.MM(false,true) -- u^T * M1
  U_MI_2 = nn.MM(false,true) -- u^T * M2

  THETA_MASK = nn.Sequential()
    :add(nn.Select(2,1))
    :add(nn.Log())
    :add(nn.Replicate(T,1))
    :add(nn.Padding(2,-1,2,-math.huge))
    :add(nn.View(1,T,nstates))

  THETA_PRE = nn.Sequential()
    :add(nn.JoinTable(1))
    :add(nn.View(T,opt.max_history))
    :add(nn.Padding(2,-1,2,-math.huge))
    :add(nn.View(1,T,nstates))

  THETA = nn.CAddTable()

  M2_T_M1 = nn.MM(true, false)
  U2 = nn.Sequential():add(nn.MM(false,true)):add(nn.View(1,D0))

  CRF = nn.Sequential():add(nn.NaN(nn.CRF(nstates))):add(nn.Exp())

  create_crf_infer_module()

  -- nngraph based model
  create_embedding_nodes()

  m2_t_m1 = M2_T_M1({c,m}):annotate({name = 'm2tm1', description = 'M2^T * M1'})
  u2 = U2({m2_t_m1,u}):annotate({name = 'u2', description = 'second layer query embeddings'})

  umi_1 = U_MI_1({u,m}):annotate({name = 'umi1', description = 'u1^T mi'})
  umi_2 = U_MI_2({u2,c}):annotate({name = 'umi2', description = 'u2^T mi'})

  theta_mask = THETA_MASK(te_inp):annotate({name = 'theta_mask', description = 'binary potential theta'})
  theta_pre = THETA_PRE({umi_1,umi_2}):annotate({name = 'theta_pre', description = 'binary potential theta'})
  theta = THETA({theta_pre,theta_mask}):annotate({name = 'theta', description = 'binary potential theta'})

  crf = CRF(theta):annotate({name = 'CRF', description = 'CRF layer'})

  create_crf_infer_node()

  model = nn.gModule({x_inp,q_inp,te_inp},{a_hat})
end

-- create full distribution nngraph modules
function create_crf_infer_module()
  -- the following computes probability distribution over all sequences [x_i, x_j, x_k]
  -- conditional probabilities p(x_j | x_i) = p(x_i,x_j) / sum_j' p(x_i,x_j')
  CP = nn.Sequential():add(nn.View(T+1,nstates,nstates)):add(nn.SplitTable(1))
    :add(nn.MapTable():add(nn.Normalize(1)))
  -- p(x_i | <nil>)
  CP1 = nn.Sequential():add(nn.SelectTable(1)):add(nn.Select(1,1)):add(nn.Replicate(nstates,2))
  -- p(x_j | x_i)
  CP2 = nn.SelectTable(2)
  -- p(<nil>, x_i, x_j) = p(x_j | x_i) * p(x_i | <nil>)
  CP12 = nn.CMulTable()
  -- p(<nil> | x_j)
  CP3 = nn.Sequential():add(nn.SelectTable(3)):add(nn.Select(2,1))
    :add(nn.Replicate(nstates,1))
  -- p(<nil>, x_i, x_j, <nil>), size = C x C
  CP123 = nn.Sequential():add(nn.CMulTable()):add(nn.View(1, nstates * nstates))

  -- compute c_i + c_j
  -- pad c with an extra row for <nil> sentence
  CPAD1 = nn.Padding(1,-1,2,0)
  CPAD2 = nn.Padding(1,-1,2,0)
  
  -- replicate c in two different ways so that summing them up would
  -- result in the desired sum
  CEMB1 = nn.Sequential():add(nn.Replicate(nstates, 1))
    :add(nn.Reshape(nstates * nstates, D0))
  CEMB2 = nn.Sequential():add(nn.SplitTable(1)):add(nn.MapTable()
    :add(nn.Replicate(nstates,1)))
    :add(nn.JoinTable(1)):add(nn.Reshape(nstates * nstates, D0))
  CEMB = nn.Sequential():add(nn.ParallelTable():add(CEMB2):add(CEMB1))
    :add(nn.CAddTable())

  O = nn.MM(false,false)
  W = nn.Sequential():add(nn.CAddTable()):add(nn.Linear(D0,nwords,false))
    :add(nn.LogSoftMax()):add(nn.Squeeze())
end

function create_crf_infer_node()
  cp = CP(crf):annotate({name = 'cp', description = 'conditional probabilities p(j | i)'})
  cp1 = CP1(cp):annotate({name = 'cp1', description = 'p(x_i | <nil>)'})
  cp2 = CP2(cp):annotate({name = 'cp2', description = 'p(x_j | x_i)'})
  cp3 = CP3(cp):annotate({name = 'cp3', description = 'p(x_k | x_j)'})
  cp12 = CP12({cp1,cp2}):annotate({name = 'cp12', description = 'p(<nil>, x_i, x_j)'})
  cp123 = CP123({cp12,cp3}):annotate({name = 'cp123', description = 'p(<nil>, x_i, x_j, x_k)'})

  cpad1 = CPAD1(c2):annotate({name = 'cpad1', description = 'nil-padded output embedding'})
  cpad2 = CPAD2(c3):annotate({name = 'cpad2', description = 'nil-padded output embedding'})
  cemb = CEMB({cpad1,cpad2}):annotate({name = 'cemb', description = 'Sum of embeddings c_i + c_j + c_k'})
  o = O({cp123,cemb}):annotate({name = 'output', description = 'Output vector'})

  a_hat = W({o,u2}):annotate({name = 'a_hat', description = 'output predictions'})

  crit = nn.ClassNLLCriterion()
end

function create_embedding_layers()
  ltx = nn.LookupTable(nwords,D0)
  ltq = ltx:clone('weight', 'gradWeight')
  A = nn.Sequential()
  B = nn.Sequential()

  if opt.pe then -- position encoding
    A_pe = nn.CMul(opt.max_history,opt.max_sent_len,D0)
    A:add(A_pe)
  end
  A:add(nn.Sum(2))
  if opt.te then -- temporal encoding
    A:add(nn.View(1,D0*opt.max_history)):add(nn.Add(D0*opt.max_history))
      :add(nn.View(opt.max_history,D0))
  end

  -- for query
  -- adding position encoding
  if opt.pe then
    B_pe = nn.CMul(train_questions:size(2),D0)
    B:add(B_pe)
  end
  B:add(nn.Sum(1)):add(nn.View(1,D0))

  -- for output representation of the memories
  C = nn.Sequential():add(nn.LookupTable(nwords,D0))
  if opt.pe then -- position encoding
    C_pe = nn.CMul(opt.max_history,opt.max_sent_len,D0)
    C:add(C_pe)
  end
  C:add(nn.Sum(2))
  if opt.te then -- temporal encoding
    C:add(nn.View(1,D0*opt.max_history)):add(nn.Add(D0*opt.max_history))
      :add(nn.View(opt.max_history,D0))
  end
  
  C2 = C:clone()
  C3 = C:clone()
  if opt.pe then
    C2_pe = C2:get(2)
    C3_pe = C3:get(2)
  end

  A_mask  = nn.CMulTable()
  C_mask  = nn.CMulTable()
  C2_mask = nn.CMulTable()
  C3_mask = nn.CMulTable()

  te_mask = torch.ones(opt.max_history, D0)
end

function create_embedding_nodes()
  x_inp = nn.Identity()():annotate({name = 'x', description = 'memories'})
  q_inp = nn.Identity()():annotate({name = 'q', description  = 'query'})
  te_inp = nn.Identity()():annotate({name = 'te_inp', description = 'temporal encoding'})

  x_pre = ltx(x_inp):annotate({name = 'x_pre', description = 'pre embeddings'})
  q_pre = ltq(q_inp):annotate({name = 'q_pre', description = 'pre embeddings'})
  m_pre = A(x_pre):annotate({name = 'm_pre', description = 'memory embeddings'})
  u = B(q_pre):annotate({name = 'u', description = 'query embeddings'})
  c_pre  = C(x_inp):annotate({name = 'c_pre', description = 'output embeddings'})
  c2_pre = C2(x_inp):annotate({name = 'c2_pre', description = 'output embeddings'})
  c3_pre = C3(x_inp):annotate({name = 'c3_pre', description = 'output embeddings'})

  m  = A_mask({m_pre,te_inp}):annotate({name = 'm', description = 'm'})
  c  = C_mask({c_pre,te_inp}):annotate({name = 'c', description = 'c'})
  c2 = C2_mask({c2_pre,te_inp}):annotate({name = 'c2', description = 'c2'})
  c3 = C3_mask({c3_pre,te_inp}):annotate({name = 'c3', description = 'c3'})
end

function adjustGrad()
  if opt.grad_norm == 'off' then
    return
  end
  local grad
  if opt.grad_norm == 'global' then
    renorm(gradParams, opt.max_grad_norm)
  elseif opt.grad_norm == 'local' then
    for i,node in ipairs(model.forwardnodes) do
      if node.data.module then
        local lmod = node.data.module
        if lmod.gradWeight and lmod.gradBias then
          grad = nn.JoinTable(1):forward({lmod.gradWeight:view(-1,1), lmod.gradBias:view(-1,1)})
        elseif lmod.gradWeight then
          grad = lmod.gradWeight:view(-1,1)
        elseif lmod.gradBias then
          grad = lmod.gradBias:view(-1,1)
        end
        renorm(grad, opt.max_grad_norm)
      end
    end
  end
  if opt.unit_test then -- test grad renorm
    newParams, newGradParams = model:getParameters()
    assert(torch.sum(gradParams:eq(newGradParams)) == gradParams:size()[1], 'renorm failed test')
    params = newParams
    gradParams = newGradParams
  end
end

function renorm(t, norm)
  if t and #t:size() > 0 then
    local t_norm = torch.sqrt(t:dot(t))
    local shrinkage = norm / t_norm
    if shrinkage < 1 then
      t:mul(shrinkage)
    end
  end
end

function adjustEta(eta, epoch)
  if epoch <= 100 then
    if epoch % 25 == 0 then
      return eta / 2
    end
  end
  return eta
end

function debugModel()
  local nanswers = test_answers:size(1)
  local all_predictions = torch.zeros(nanswers)
  local all_prediction_scores = torch.zeros(nanswers, nwords)
  local all_marginals

  debugFile = opt.debug
  proto = torch.load(opt.debug)
  model = proto.model
  opt = proto.options
  max_history = opt.max_history
  D0 = opt.D0
  max_sent_len = opt.max_sent_len
  nstates = max_history + 1
  T = 2
  te_mask = torch.ones(max_history, D0)

  local outDebugFile = hdf5.open(''..debugFile..'.debug', 'w')

  if string.find(debugFile, ".binarycrf") or string.find(debugFile, ".unarycrf") then
    print('Detected crf model')

    if string.find(debugFile, ".binarycrf") then
      theta0 = torch.zeros(1, nstates, nstates)
      theta12 = torch.zeros(1, nstates, nstates)
      theta3 = torch.zeros(1, nstates, nstates)
    end

    -- data to save to file
    all_marginals = torch.zeros(nanswers, T + 1, nstates, nstates)
    local all_distribution = torch.zeros(nanswers, T, nstates - 1)

    for i = 1, test_stories:size(1) do
      local x = getStory(test_stories, i, max_history, max_sent_len)
      local q = test_questions[i]
      all_prediction_scores[i] = model:forward({x,q,te_mask,theta0,theta12,theta3})
      _, all_predictions[i] = torch.max(all_prediction_scores[i]:float(), 1)
      for _, p in ipairs(model.forwardnodes) do
        if p.data.annotations.name == 'nom' then
          all_distribution[i] = p.data.module.output:view(T,nstates-1)
        elseif p.data.annotations.name == 'CRF' then
          all_marginals[i] = p.data.module.output
        end
      end
    end
    outDebugFile:write('distribution', all_distribution)
  end

  print('Accuracy = '..torch.eq(all_predictions:long(), test_answers):sum()/nanswers)
  outDebugFile:write('scores', all_prediction_scores)
  outDebugFile:write('answers', test_answers:squeeze())
  outDebugFile:write('predictions', all_predictions:long())
  outDebugFile:write('marginals', all_marginals)
  outDebugFile:close()
end

function testModel(useHeldout)
  if useHeldout then
    testX = heldout_stories
    testQ = heldout_questions
    testA = heldout_answers
    accuracyText = "Accuracy on held out = "
  else
    testX = test_stories
    testQ = test_questions
    testA = test_answers
    accuracyText = "Accuracy on test set = "
    model = bestHeldoutModel
  end

  if opt.cuda then
    testX = testX:cuda()
    testQ = testQ:cuda()
  end

  local Y_hat = torch.zeros(testA:size(1))
  zeroLookupTable()
  for i=1, testX:size(1) do
    local x = getStory(testX,i,opt.max_history, opt.max_sent_len)
    local q = testQ[i]
    local preds = model:forward({x,q,te_mask,theta0,theta12,theta3})
    _, Y_hat[i] = torch.max(preds:float(),1)
  end
  local correct = torch.eq(Y_hat:long() - testA, 0):sum()
  local accuracy = correct/Y_hat:size(1)
  print(accuracyText..accuracy)

  if useHeldout then
    if bestHeldoutAccuracy < accuracy then
      bestHeldoutAccuracy = accuracy
      bestHeldoutModel = model:clone()
    end
  else
    -- re-test the best model on held out set for sanity check
    local Y_hat_heldout = torch.zeros(heldout_answers:size(1))
    for i=1, heldout_stories:size(1) do
      local x = getStory(heldout_stories,i,opt.max_history, opt.max_sent_len)
      local q = heldout_questions[i]
      local preds = model:forward({x,q,te_mask,theta0,theta12,theta3})
      _, Y_hat_heldout[i] = torch.max(preds:float(),1)
    end
    local accuracy_heldout = torch.eq(Y_hat_heldout:long()-heldout_answers, 0):sum() / Y_hat_heldout:size(1)
    if accuracy_heldout ~= bestHeldoutAccuracy then
      print('Best model on held out set is lost, cannot reproduce accuracy '..
        bestHeldoutAccuracy .. ', actual accuracy = ' .. accuracy_heldout)
    else
      print('Using model which achieved ' .. accuracy_heldout .. ' on held out set.')
    end
    if opt.save and accuracy >= opt.saveminacc then
      local acc = torch.LongTensor({accuracy*10000}):double()[1]/100
      local modelFile = opt.datafile.."."..acc.."."..opt.classifier
      torch.save(modelFile, {model = model, options = opt})
      print('Saved model to ' .. modelFile)
    end
  end
end

function makePosEncMat(inputLayer)
  if inputLayer == nil then
    return
  end
  
  local input = inputLayer.weight
  input:zero()

  if input:dim() == 3 then
    num_sent , sent_len, embed_size = input:size(1), input:size(2), input:size(3)
    for i=1, num_sent do
      for j=1, sent_len do
        for k=1, embed_size do
          input[i][j][k] = (1-j/sent_len) - (k/embed_size)*(1- (2*j/sent_len))
        end
      end
    end
  else
    sent_len, embed_size = input:size(1), input:size(2)
    for j=1, sent_len do
      for k=1, embed_size do
        input[j][k] = (1-j/sent_len) - (k/embed_size)*(1- (2*j/sent_len))
      end
    end
  end

end

-- convert 1 x all_stories_by_question to num_stories x max_single_story_len
function getStory(X,q_id,max_history,max_sent_len)
  local story = X[ {q_id, {num_history - max_history + 1, num_history}, {1, max_sent_len} } ]
  
  -- detect empty memories and clear out theta potentials
  local num_empty_sentences = torch.sum(story[{{},1}]:eq(idx_pad))

  if te_mask then
    te_mask:fill(1)
    if num_empty_sentences > 0 then
      te_mask[{{1,num_empty_sentences}}]:fill(0)
    end
  end

  num_empty_sentences = num_empty_sentences + 1 -- add 1 for the <nil> state

  if theta0 then
    theta0:fill(-math.huge)
    theta0[{{},1,{num_empty_sentences+1,nstates}}]:fill(0) -- first row
  end

  if theta3 then
    theta3:fill(-math.huge)
    theta3[{{},{num_empty_sentences+1,nstates},1}]:fill(0) -- first column
  end

  if theta12 then
    theta12:fill(-math.huge)
    theta12[{{},{num_empty_sentences+1,nstates},{num_empty_sentences+1,nstates}}]:fill(0) -- bottom right n x n square
    for i=1,nstates do
      theta12[{1,i,i}] = -math.huge
    end
  end

  return story
end

function get_module_output(name)
  for i,v in ipairs(model.forwardnodes) do
    if v.data.annotations.name == name and v.data.module then
      return v.data.module.output
    end
  end
  return nil
end

function n_zero_rows(ts)
  if ts then
    return torch.sum(torch.sum(torch.abs(ts),2):eq(0))
  else
    return 0
  end
end

function resetPE()
  makePosEncMat(A_pe)
  makePosEncMat(B_pe)
  makePosEncMat(C_pe)
  makePosEncMat(C2_pe)
  makePosEncMat(C3_pe)
end

function zeroPEGrad()
  if opt.pe then
    if A_pe then A_pe.gradWeight:zero() end
    if B_pe then B_pe.gradWeight:zero() end
    if C_pe then C_pe.gradWeight:zero() end
    if C2_pe then C2_pe.gradWeight:zero() end
    if C3_pe then C3_pe.gradWeight:zero() end
  end
end

function zeroLookupTable()
  zeroWeight(ltx.weight)
  zeroWeight(ltq.weight)
  if ltn then
    zeroWeight(ltn.weight)
  end
  if C2 then zeroWeight(C2.modules[1].weight) end
  if C3 then zeroWeight(C3.modules[1].weight) end
  zeroWeight(C.modules[1].weight)
end

function zeroWeight(wt)
  wt[idx_pad]:zero()
  wt[idx_start]:zero()
  wt[idx_end]:zero()
  wt[idx_rare]:zero()
end

function Set (list)
  local set = {}
  for _, l in ipairs(list) do set[l] = true end
  return set
end

function load()
  if opt.debug ~= '' then
    opt.datafile = string.sub(opt.debug, 1, 9)
    print('Loading detected data file ' .. opt.datafile)
  end
  -- get the data out of the datafile
  local f = hdf5.open(opt.datafile, 'r')
  local data = f:all()

  idx_start = data.idx_start[1]
  idx_end   = data.idx_end[1]
  idx_pad   = data.idx_pad[1]
  idx_rare  = data.idx_rare[1]

  nwords = data.nwords[1]

  train_stories   = data.train_stories:long() -- [# Questions x Max Story Length]
  train_questions = data.train_questions:long() -- [# Questions x Max Q Length]
  train_answers   = data.train_answers:long() -- [# Questions x 1]
  train_facts     = data.train_facts:long() -- [# Questions x Max Fact Length]

  local ntrains = train_stories:size(1)
  local endtrain = math.floor(ntrains * 0.9)

  heldout_stories = train_stories[{ {endtrain + 1, ntrains} }]
  heldout_questions = train_questions[{ {endtrain + 1, ntrains} }]
  heldout_answers = train_answers[{ {endtrain + 1, ntrains} }]
  heldout_facts = train_facts[{ {endtrain + 1, ntrains} }]

  train_stories = train_stories[{ {1, endtrain} }]
  train_questions = train_questions[{ {1, endtrain} }]
  train_answers = train_answers[{ {1, endtrain} }]
  train_facts = train_facts[{ {1, endtrain} }]  

  test_stories   = data.test_stories:long()
  test_questions = data.test_questions:long()
  test_answers   = data.test_answers:long()
  test_facts     = data.test_facts:long()

  num_history = train_stories:size(2)
  len_sentence = train_stories:size(3)

  opt.max_history = math.min(opt.max_history, num_history)
  opt.max_sent_len = math.min(opt.max_sent_len, len_sentence)

  bestHeldoutAccuracy = 0
end


main()
