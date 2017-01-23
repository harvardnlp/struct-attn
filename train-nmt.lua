require 'nn'
require 'nngraph'

require 'hdf5'
require 'data-nmt.lua'
require 'models/models-nmt.lua'
require 'models/model_utils.lua'

cmd = torch.CmdLine()

-- data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file','data/nmt-train.hdf5', [[Path to the training *.hdf5 file from preprocess.py]])
cmd:option('-val_data_file','data/nmt-val.hdf5', [[Path to validation *.hdf5 file from preprocess.py]])
cmd:option('-savefile', 'nmt', [[Savefile name (model will be saved as
                                             savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is
                                             the validation perplexity]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])

-- rnn model specs
cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
cmd:option('-attn', 'softmax', [[Attention type: one of {softmax, sigmoid, crf}]])
cmd:option('-lambda', 2, '[[Normalization lambda for marginals if using structured (i.e. crf) attention]]')
cmd:option('-lambda2', 0.005, '[[L2 penalty for CRF bias terms for structured attention]]')

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")
-- optimization
cmd:option('-epochs', 30, [[Number of training epochs]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd (vanilla SGD), adagrad, adadelta, adam]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings: sgd =1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1]])
cmd:option('-max_grad_norm', 1, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-start_decay', 0, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])

cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                             on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-max_batch_l', 128, [[If blank, then it will infer the max batch size from validation
                               data. You should only use this if your validation set uses a different
                               batch size in the preprocessing step]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")
cmd:option('-start_symbol', 0, [[Use special start-of-sentence and end-of-sentence tokens
                               on the source side. We've found this to make minimal difference]])
-- GPU
cmd:option('-gpuid', 1, [[Which gpu to use. -1 = use CPU]])
-- bookkeeping
cmd:option('-print_every', 100, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-prealloc', 1, [[Use memory preallocation and sharing between cloned encoder/decoders]])
function zero_table(t)
  for i = 1, #t do
    t[i]:zero()
  end
end

function train(train_data, valid_data)

  local timer = torch.Timer()
  local num_params = 0
  local start_decay = 0
  params, grad_params = {}, {}
  if opt.train_from:len() == 0 then
    opt.train_perf = {}
    opt.val_perf = {}
  end
  for i = 1, #layers do
    local p, gp = layers[i]:getParameters()
    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)
    end
    num_params = num_params + p:size(1)
    params[i] = p
    grad_params[i] = gp
  end
  print("Number of parameters: " .. num_params)
  word_vec_layers[1].weight[1]:zero()
  word_vec_layers[2].weight[1]:zero()

  -- prototypes for gradients so there is no need to clone
  context_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  context_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  -- clone encoder/decoder up to max source/target length
  decoder_clones = clone_many_times(decoder, opt.max_sent_l_targ)
  encoder_clones = clone_many_times(encoder, opt.max_sent_l_src)
  for i = 1, opt.max_sent_l_src do
    if encoder_clones[i].apply then
      encoder_clones[i]:apply(function(m) m:setReuse() end)
      if opt.prealloc == 1 then encoder_clones[i]:apply(function(m) m:setPrealloc() end) end
    end
  end
  for i = 1, opt.max_sent_l_targ do
    if decoder_clones[i].apply then
      decoder_clones[i]:apply(function(m) m:setReuse() end)
      if opt.prealloc == 1 then decoder_clones[i]:apply(function(m) m:setPrealloc() end) end
    end
  end
  
  local h_init = torch.zeros(opt.max_batch_l, opt.rnn_size)
  if opt.gpuid >= 0 then
    h_init = h_init:cuda()
    cutorch.setDevice(opt.gpuid)
    context_proto = context_proto:cuda()
    context_grad_proto = context_grad_proto:cuda()
  end

  -- these are initial states of encoder/decoder for fwd/bwd steps
  init_layer = {}
  for L = 1, opt.num_layers do
    table.insert(init_layer, h_init:clone())
    table.insert(init_layer, h_init:clone())    
  end
  
  function reset_state(state, batch_l, t)
    if t == nil then
      local u = {}
      for i = 1, #state do
        state[i]:zero()
        table.insert(u, state[i][{{1, batch_l}}])
      end
      return u
    else
      local u = {[t] = {}}
      for i = 1, #state do
        state[i]:zero()
        table.insert(u[t], state[i][{{1, batch_l}}])
      end
      return u
    end
  end
  -- decay learning rate if val perf does not improve or we hit the opt.start_decay_at limit
  function decay_lr(epoch)
    print(opt.val_perf)
    if opt.val_perf[#opt.val_perf] ~= nil and opt.val_perf[#opt.val_perf-1] ~= nil then
      local curr_ppl = opt.val_perf[#opt.val_perf]
      local prev_ppl = opt.val_perf[#opt.val_perf-1]
      if curr_ppl > prev_ppl then
	opt.start_decay = 1
      else
	local savefile = string.format('%s.t7', opt.savefile)
	print('saving ' .. savefile)
	torch.save(savefile, {layers, opt})
      end
    else
      local savefile = string.format('%s.t7', opt.savefile)
      print('saving ' .. savefile)
      torch.save(savefile, {layers, opt})
    end
    if opt.start_decay == 1 then
      opt.learning_rate = opt.learning_rate * opt.lr_decay
    end    
  end

  function train_batch(data, epoch)
    local train_nonzeros = 0
    local train_loss = 0
    local batch_order = torch.randperm(data.length) -- shuffle mini batch order
    local start_time = timer:time().real
    local num_words_target = 0
    local num_words_source = 0
    for i = 1, data:size() do
      zero_table(grad_params, 'zero')
      local d = data[batch_order[i]]
      local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
      local batch_l, target_l, source_l = d[5], d[6], d[7]
      local context = context_proto[{{1, batch_l}, {1, source_l}}]
      local context_grads = context_grad_proto[{{1, batch_l}, {1, source_l}}]:zero()
      local rnn_state_enc = reset_state(init_layer, batch_l, 0)      
      local encoder_inputs = {}
      -- forward prop encoder
      for t = 1, source_l do
        encoder_clones[t]:training()
        encoder_inputs[t] = {source[t], table.unpack(rnn_state_enc[t-1])}
        rnn_state_enc[t] = encoder_clones[t]:forward(encoder_inputs[t])
        context[{{},t}]:copy(rnn_state_enc[t][#rnn_state_enc[t]])
      end

      -- copy encoder last hidden state to decoder initial state
      local rnn_state_dec = reset_state(init_layer, batch_l, 0)
      -- forward prop decoder
      local decoder_inputs = {}
      for t = 1, target_l do
        decoder_clones[t]:training()	
        decoder_inputs[t] = {target[t], table.unpack(rnn_state_dec[t-1])}
	rnn_state_dec[t] = decoder_clones[t]:forward(decoder_inputs[t])
      end
      if opt.attn == 'crf' then
	all_layers.crf.gradWeight:add(all_layers.crf.weight):mul(opt.lambda2)
      end      
      -- backward prop decoder
      local drnn_state_dec = reset_state(init_layer, batch_l)
      local loss = 0
      for t = target_l, 1, -1 do
	local generator_input = {context, rnn_state_dec[t][#rnn_state_dec[t]]}
        local pred = generator:forward(generator_input)
        loss = loss + criterion:forward(pred, target_out[t])/batch_l
        local dl_dpred = criterion:backward(pred, target_out[t])
        dl_dpred:div(batch_l)
        local dl_dtarget = generator:backward(generator_input, dl_dpred)
        drnn_state_dec[#drnn_state_dec]:add(dl_dtarget[2])
	context_grads:add(dl_dtarget[1])
        local dlst = decoder_clones[t]:backward(decoder_inputs[t], drnn_state_dec)
        for j = 1, #drnn_state_dec do
          drnn_state_dec[j]:copy(dlst[j+1])
        end	
      end
      word_vec_layers[2].gradWeight[1]:zero()
      -- backward prop encoder
      local drnn_state_enc = reset_state(init_layer, batch_l)
      for t = source_l, 1, -1 do
	drnn_state_enc[#drnn_state_enc]:add(context_grads[{{},t}])
        local dlst = encoder_clones[t]:backward(encoder_inputs[t], drnn_state_enc)
        for j = 1, #drnn_state_enc do
          drnn_state_enc[j]:copy(dlst[j+1])
        end
      end
      word_vec_layers[1].gradWeight[1]:zero()
      local grad_norm = 0
      for j = 1, #grad_params do
	grad_norm = grad_norm + grad_params[j]:norm()^2
      end
      grad_norm = grad_norm^0.5
      -- Shrink norm and update params
      local param_norm = 0
      local shrinkage = opt.max_grad_norm / grad_norm
      for j = 1, #grad_params do
	if shrinkage < 1 then
	  grad_params[j]:mul(shrinkage)
	end
	if opt.optim == 'adagrad' then
	  adagrad_step(params[j], grad_params[j], layer_etas[j], optStates[j])
	elseif opt.optim == 'adadelta' then
	  adadelta_step(params[j], grad_params[j], layer_etas[j], optStates[j])
	elseif opt.optim == 'adam' then
	  adam_step(params[j], grad_params[j], layer_etas[j], optStates[j])
	else
	  params[j]:add(grad_params[j]:mul(-opt.learning_rate))
	end
	param_norm = param_norm + params[j]:norm()^2
      end
      param_norm = param_norm^0.5
      -- Bookkeeping
      num_words_target = num_words_target + batch_l*target_l
      num_words_source = num_words_source + batch_l*source_l
      train_nonzeros = train_nonzeros + nonzeros
      train_loss = train_loss + loss*batch_l
      local time_taken = timer:time().real - start_time
      if i % opt.print_every == 0 then
        local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
          epoch, i, data:size(), batch_l, opt.learning_rate)
        stats = stats .. string.format('PPL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
          math.exp(train_loss/train_nonzeros), param_norm, grad_norm)
        stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
          (num_words_target+num_words_source) / time_taken,
          num_words_source / time_taken,
          num_words_target / time_taken)
        print(stats)
      end
      if i % 200 == 0 then
        collectgarbage()
      end
    end
    return train_loss, train_nonzeros
  end

  local total_loss, total_nonzeros, batch_loss, batch_nonzeros
  for epoch = 1, opt.epochs do
    generator:training()
    total_loss, total_nonzeros = train_batch(train_data, epoch)
    local train_score = math.exp(total_loss/total_nonzeros)
    print('Train', train_score)
    opt.train_perf[#opt.train_perf + 1] = train_score
    local score = eval(valid_data)
    opt.val_perf[#opt.val_perf + 1] = score
    if opt.optim == 'sgd' then --only decay with SGD
      decay_lr(epoch)
    end
  end
  -- save final model
  local savefile = string.format('%s_final.t7', opt.savefile)
  print('saving final model to ' .. savefile)
  torch.save(savefile, {{encoder:double(), decoder:double(), generator:double()}, opt})
end

function eval(data)
  generator:evaluate()  
  local nll = 0
  local total = 0
  for i = 1, data:size() do
    local d = data[i]
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l = d[5], d[6], d[7]
    local rnn_state_enc = reset_state(init_layer, batch_l, 0)
    local context = context_proto[{{1, batch_l}, {1, source_l}}]
    local encoder_inputs = {}
    -- forward prop encoder
    for t = 1, source_l do
      encoder_clones[t]:evaluate()
      encoder_inputs[t] = {source[t], table.unpack(rnn_state_enc[t-1])}
      rnn_state_enc[t] = encoder_clones[t]:forward(encoder_inputs[t])
      context[{{},t}]:copy(rnn_state_enc[t][#rnn_state_enc[t]])
    end

    local rnn_state_dec = reset_state(init_layer, batch_l, 0)
    -- forward prop decoder
    local decoder_inputs = {}
    for t = 1, target_l do
      decoder_clones[t]:evaluate()	
      decoder_inputs[t] = {target[t], table.unpack(rnn_state_dec[t-1])}
      rnn_state_dec[t] = decoder_clones[t]:forward(decoder_inputs[t])
      local generator_input = {context, rnn_state_dec[t][#rnn_state_dec[t]]}
      local pred = generator:forward(generator_input)
      nll = nll + criterion:forward(pred, target_out[t])      
    end
    total = total + nonzeros
  end
  local valid = math.exp(nll / total)
  print("Valid", valid)
  collectgarbage()
  return valid
end

function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vecs_dec' then
      table.insert(word_vec_layers, layer)
    elseif layer.name == 'word_vecs_enc' then
      table.insert(word_vec_layers, layer)
    else      
      all_layers[layer.name] = layer
    end
  end
end

function main()
  -- parse input params
  opt = cmd:parse(arg)

  torch.manualSeed(opt.seed)

  if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)
  end
  
  -- Create the data loader class.
  print('loading data...')
  train_data = data.new(opt, opt.data_file)
  valid_data = data.new(opt, opt.val_data_file)
  print('done!')
  print(string.format('Source vocab size: %d, Target vocab size: %d',
      valid_data.source_size, valid_data.target_size))
  opt.max_sent_l_src = valid_data.source:size(2)
  opt.max_sent_l_targ = valid_data.target:size(2)
  opt.max_sent_l = math.max(opt.max_sent_l_src, opt.max_sent_l_targ)
  if opt.max_batch_l == '' then
    opt.max_batch_l = valid_data.batch_l:max()
  end

  print(string.format('Source max sent len: %d, Target max sent len: %d',
      valid_data.source:size(2), valid_data.target:size(2)))

  -- Enable memory preallocation - see memory.lua
  preallocateMemory(opt.prealloc)
  opt.context_size = opt.rnn_size
  -- Build model
  if opt.train_from:len() == 0 then
    encoder = make_lstm(valid_data, opt, 'enc')
    decoder = make_lstm(valid_data, opt, 'dec')
    generator = make_generator(valid_data, opt)
  else
    assert(path.exists(opt.train_from), 'checkpoint path invalid')
    print('loading ' .. opt.train_from .. '...')
    local checkpoint = torch.load(opt.train_from)
    local model, model_opt = checkpoint[1], checkpoint[2]
    print(model_opt)
    opt.val_perf = model_opt.val_perf
    opt.train_perf = model_opt.train_perf
    opt.num_layers = model_opt.num_layers
    opt.rnn_size = model_opt.rnn_size
    encoder = model[1]:double()
    decoder = model[2]:double()
    generator = model[3]:double()
  end
  local w = torch.ones(valid_data.target_size)
  w[1] = 0
  criterion = nn.ClassNLLCriterion(w)
  criterion.sizeAverage = false
  layers = {encoder, decoder, generator}
  if opt.optim ~= 'sgd' then
    layer_etas = {}
    optStates = {}
    for i = 1, #layers do
      layer_etas[i] = opt.learning_rate -- can have layer-specific lr, if desired
      optStates[i] = {}
    end
  end

  if opt.gpuid >= 0 then
    for i = 1, #layers do
      layers[i]:cuda()
    end
    criterion:cuda()
  end

  -- these layers will be manipulated during training
  word_vec_layers = {}
  all_layers = {}
  encoder:apply(get_layer)
  decoder:apply(get_layer)  
  generator:apply(get_layer)  
  all_layers.attn_layer:apply(get_layer)
  train(train_data, valid_data)
end

main()
