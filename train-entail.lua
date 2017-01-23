require 'nn'
require 'nngraph'
require 'hdf5'

require 'data-entail.lua'
require 'models/models-entail.lua'
require 'models/model_utils.lua'

cmd = torch.CmdLine()

-- data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file','data/entail-train.hdf5', [[Path to the training *.hdf5 file]])
cmd:option('-val_data_file', 'data/entail-val.hdf5', [[Path to validation *.hdf5 file]])
cmd:option('-test_data_file','data/entail-test.hdf5',[[Path to test *.hdf5 file]])

cmd:option('-savefile', 'entail', [[Savefile name]])

-- model specs
cmd:option('-hidden_size', 300, [[MLP hidden layer size]])
cmd:option('-word_vec_size', 300, [[Word embedding size]])
cmd:option('-attn', 'none', [[one of {none, simple, struct}.
                              none = no intra-sentence attention (baseline model)
                              simple = simple attention model
                              struct = structured attention (syntactic attention)]])
cmd:option('-num_layers_parser', 1, [[Number of layers for the RNN parsing layer]])
cmd:option('-rnn_size_parser', 100, [[size of the RNN for the parsing layer]])
cmd:option('-use_parent', 1, [[Use soft parents]])
cmd:option('-use_children', 0, [[Use soft children]])
cmd:option('-share_params',1, [[Share parameters between the two sentence encoders]])
cmd:option('-proj', 1, [[Have a projection layer from the Glove embeddings]])
cmd:option('-dropout', 0.2, [[Dropout probability.]])   

-- optimization
cmd:option('-epochs', 100, [[Number of training epochs]])
cmd:option('-param_init', 0.01, [[Parameters are initialized over uniform distribution with support
                               (-param_init, param_init)]])
cmd:option('-optim', 'adagrad', [[Optimization method. Possible options are: 
                              sgd (vanilla SGD), adagrad, adadelta, adam]])
  cmd:option('-learning_rate', 0.05, [[Starting learning rate. If adagrad/adadelta/adam is used, 
                                then this is the global learning rate.]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it
                               to have the norm equal to max_grad_norm]])
cmd:option('-pre_word_vecs', 'data/glove.hdf5', [[If a valid path is specified, then this will load 
                                      pretrained word embeddings (hdf5 file)]])
cmd:option('-fix_word_vecs', 1, [[If = 1, fix word embeddings]])
cmd:option('-max_batch_l', 32, [[If blank, then it will infer the max batch size from validation 
				   data. You should only use this if your validation set uses a different
				   batch size in the preprocessing step]])
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-print_every', 1000, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

function zero_table(t)
  for i = 1, #t do
    t[i]:zero()
  end
end

function train(train_data, valid_data)

  local timer = torch.Timer()
  local start_decay = 0
  params, grad_params = {}, {}
  opt.train_perf = {}
  opt.val_perf = {}
  
  for i = 1, #layers do
    local p, gp = layers[i]:getParameters()
    local rand_vec = torch.randn(p:size(1)):mul(opt.param_init)
    if opt.gpuid >= 0 then
      rand_vec = rand_vec:cuda()
    end	 
    p:copy(rand_vec)	 
    params[i] = p
    grad_params[i] = gp
  end
  if opt.pre_word_vecs:len() > 0 then
    print("loading pre-trained word vectors")
    local f = hdf5.open(opt.pre_word_vecs)     
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      word_vecs_enc1.weight[i]:copy(pre_word_vecs[i])
      word_vecs_enc2.weight[i]:copy(pre_word_vecs[i])       
    end
  end

  --copy shared params   
  params[2]:copy(params[1])   
  if opt.attn ~= 'none' then
    params[7]:copy(params[6])
  end

  if opt.share_params == 1 then
    if opt.proj == 1 then
      entail_layers.proj2.weight:copy(entail_layers.proj1.weight)
    end      
    for k = 2, 5, 3 do	 
      entail_layers.f2.modules[k].weight:copy(entail_layers.f1.modules[k].weight)
      entail_layers.f2.modules[k].bias:copy(entail_layers.f1.modules[k].bias)
      entail_layers.g2.modules[k].weight:copy(entail_layers.g1.modules[k].weight)
      entail_layers.g2.modules[k].bias:copy(entail_layers.g1.modules[k].bias)
    end      
  end
  
  -- prototypes for gradients so there is no need to clone
  word_vecs1_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.word_vec_size)
  word_vecs2_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.word_vec_size)
  sent1_context_proto = torch.zeros(opt.max_batch_l, opt.rnn_size_parser * 2)
  sent2_context_proto = torch.zeros(opt.max_batch_l, opt.rnn_size_parser * 2)
  parser_context1_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size_parser * 2)
  parser_graph1_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.word_vec_size*2)
  parser_context2_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size_parser * 2)
  parser_graph2_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.word_vec_size*2)
  
  -- clone encoder/decoder up to max source/target length
  if opt.attn ~= 'none' then
    parser_fwd_clones = clone_many_times(parser_fwd, opt.max_sent_l_src + opt.max_sent_l_targ)
    parser_bwd_clones = clone_many_times(parser_bwd, opt.max_sent_l_src + opt.max_sent_l_targ)
    for i = 1, opt.max_sent_l_src + opt.max_sent_l_targ do
      if parser_fwd_clones[i].apply then
	parser_fwd_clones[i]:apply(function(m) m:setReuse() end)
      end      
      if parser_bwd_clones[i].apply then
	parser_bwd_clones[i]:apply(function(m) m:setReuse() end)
      end
    end      
  end         
  
  local h_init_parser = torch.zeros(opt.max_batch_l, opt.rnn_size_parser)
  if opt.gpuid >= 0 then
    h_init_parser = h_init_parser:cuda()
    cutorch.setDevice(opt.gpuid)                        
    word_vecs1_grad_proto = word_vecs1_grad_proto:cuda()
    word_vecs2_grad_proto = word_vecs2_grad_proto:cuda()
    parser_context1_proto = parser_context1_proto:cuda()
    parser_context2_proto = parser_context2_proto:cuda()            
    parser_graph1_grad_proto = parser_graph1_grad_proto:cuda()
    parser_graph2_grad_proto = parser_graph1_grad_proto:cuda()
    sent1_context_proto = sent1_context_proto:cuda()
    sent2_context_proto = sent2_context_proto:cuda()
  end

  -- these are initial states of parser/encoder/decoder for fwd/bwd steps
  init_parser = {}
  for L = 1, opt.num_layers_parser do
    table.insert(init_parser, h_init_parser:clone())
    table.insert(init_parser, h_init_parser:clone())      
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
  

  function train_batch(data, epoch)
    local train_loss = 0
    local train_sents = 0
    local batch_order = torch.randperm(data.length) -- shuffle mini batch order     
    local start_time = timer:time().real
    local num_words_target = 0
    local num_words_source = 0
    local train_num_correct = 0 
    sent_encoder:training()
    for i = 1, data:size() do
      zero_table(grad_params, 'zero')
      local d = data[batch_order[i]]
      local target, source, batch_l, target_l, source_l, label = table.unpack(d)	 
      
      -- resize the various temporary tensors that are going to hold contexts/grads
      local word_vecs1_grads = word_vecs1_grad_proto[{{1, batch_l}, {1, source_l}}]:zero()
      local word_vecs2_grads = word_vecs2_grad_proto[{{1, batch_l}, {1, target_l}}]:zero()
      local parser_context1 = parser_context1_proto[{{1, batch_l}, {1, source_l}}]
      local parser_context2 = parser_context2_proto[{{1, batch_l}, {1, target_l}}]
      local sent1_context = sent1_context_proto[{{1, batch_l}}]
      local sent2_context = sent2_context_proto[{{1, batch_l}}]
      local word_vecs1 = word_vecs_enc1:forward(source)
      local word_vecs2 = word_vecs_enc2:forward(target)	 

      if opt.attn ~= 'none' then
	set_size_encoder(batch_l, source_l, target_l, opt.word_vec_size,
			 opt.hidden_size, entail_layers)	    
	set_size_parser(batch_l, source_l, opt.rnn_size_parser*2, parser_layers1)
	set_size_parser(batch_l, target_l, opt.rnn_size_parser*2, parser_layers2)
	------ fwd prop for parser brnn for sent 1------
	-- fwd direction
	local rnn_state_parser_fwd1 = reset_state(init_parser, batch_l, 0)
	parser_fwd_inputs1 = {}	 
	for t = 1, source_l do	    
	  parser_fwd_clones[t]:training()
	  parser_fwd_inputs1[t] = {word_vecs1[{{}, t}],
				   table.unpack(rnn_state_parser_fwd1[t-1])}
	  local out = parser_fwd_clones[t]:forward(parser_fwd_inputs1[t])
	  rnn_state_parser_fwd1[t] = out
	  parser_context1[{{}, t, {1, opt.rnn_size_parser}}]:copy(out[#out])
	end
	-- bwd direction
	local rnn_state_parser_bwd1 = reset_state(init_parser, batch_l, source_l+1)
	parser_bwd_inputs1 = {}	 
	for t = source_l, 1, -1 do
	  parser_bwd_clones[t]:training()
	  parser_bwd_inputs1[t] = {word_vecs1[{{}, t}],
				   table.unpack(rnn_state_parser_bwd1[t+1])}
	  local out = parser_bwd_clones[t]:forward(parser_bwd_inputs1[t])
	  rnn_state_parser_bwd1[t] = out
	  parser_context1[{{}, t,
	      {opt.rnn_size_parser+1, opt.rnn_size_parser*2}}]:copy(out[#out])
	end
	------ fwd prop for parser brnn for sent 2------
	-- fwd direction
	local rnn_state_parser_fwd2 = reset_state(init_parser, batch_l, 0)
	parser_fwd_inputs2 = {}	 
	for t = 1, target_l do	    
	  parser_fwd_clones[t+source_l]:training()
	  parser_fwd_inputs2[t] = {word_vecs2[{{}, t}],
				   table.unpack(rnn_state_parser_fwd2[t-1])} 
	  local out = parser_fwd_clones[t+source_l]:forward(parser_fwd_inputs2[t])
	  rnn_state_parser_fwd2[t] = out
	  parser_context2[{{}, t, {1, opt.rnn_size_parser}}]:copy(out[#out])
	end
	-- bwd direction
	local rnn_state_parser_bwd2 = reset_state(init_parser, batch_l, target_l+1)
	parser_bwd_inputs2 = {}	 
	for t = target_l, 1, -1 do
	  parser_bwd_clones[t+source_l]:training()
	  parser_bwd_inputs2[t] = {word_vecs2[{{}, t}],
				   table.unpack(rnn_state_parser_bwd2[t+1])}
	  local out = parser_bwd_clones[t+source_l]:forward(parser_bwd_inputs2[t])
	  rnn_state_parser_bwd2[t] = out
	  parser_context2[{{}, t, {opt.rnn_size_parser+1,
				   opt.rnn_size_parser*2}}]:copy(out[#out])
	end
	parser_context1 = parser_context1:contiguous()
	parser_context2 = parser_context2:contiguous()
	parsed_context1 = parser_graph1:forward(parser_context1)
	parsed_context2 = parser_graph2:forward(parser_context2)
	pred_input = {word_vecs1, word_vecs2, parsed_context1, parsed_context2}
      else
	set_size_encoder(batch_l, source_l, target_l,
			 opt.word_vec_size, opt.hidden_size, entail_layers)	    	    
	pred_input = {word_vecs1, word_vecs2}
      end
      local pred_label = sent_encoder:forward(pred_input)
      local _, pred_argmax = pred_label:max(2)
      train_num_correct = train_num_correct + pred_argmax:double():view(batch_l):eq(label:double()):sum()	 
      local loss = disc_criterion:forward(pred_label, label)
      local dl_dp = disc_criterion:backward(pred_label, label)
      dl_dp:div(batch_l)

      if opt.attn ~= 'none' then
	local dl_dinput1, dl_dinput2, dl_dparser1, dl_dparser2 = table.unpack(
	  sent_encoder:backward(pred_input, dl_dp))
	------ backprop for graph-based parser ------
	parser_grads1 = parser_graph1:backward(parser_context1, dl_dparser1)
	parser_grads2 = parser_graph2:backward(parser_context2, dl_dparser2)
	word_vecs1_grads:add(dl_dinput1)
	word_vecs2_grads:add(dl_dinput2)
	
	------ backprop for parser brnn ------
	-- backprop through fwd parser rnn
	local drnn_state = reset_state(init_parser, batch_l)
	for t = source_l, 1, -1 do
	  drnn_state[#drnn_state]:add(
	    parser_grads1[{{}, t, {1, opt.rnn_size_parser}}])
	  local dlst = parser_fwd_clones[t]:backward(parser_fwd_inputs1[t], drnn_state)
	  for j = 1, #drnn_state do
	    drnn_state[j]:copy(dlst[j+1])
	  end
	  word_vecs1_grads[{{}, t}]:add(dlst[1])
	end
	-- backprop through bwd parser rnn
	local drnn_state = reset_state(init_parser, batch_l)
	for t = 1, source_l do
	  drnn_state[#drnn_state]:add(
	    parser_grads1[{{}, t, {opt.rnn_size_parser+1, 2*opt.rnn_size_parser}}])
	  local dlst = parser_bwd_clones[t]:backward(parser_bwd_inputs1[t], drnn_state)
	  for j = 1, #drnn_state do
	    drnn_state[j]:copy(dlst[j+1])
	  end
	  word_vecs1_grads[{{}, t}]:add(dlst[1])
	end
	------ backprop through source word vectors ------
	if opt.proj == 0 then
	  word_vecs_enc1:backward(source, word_vecs1_grads:contiguous())
	end	    

	------ backprop for parser brnn ------
	-- backprop through fwd parser rnn
	local drnn_state = reset_state(init_parser, batch_l)
	for t = target_l, 1, -1 do
	  drnn_state[#drnn_state]:add(
	    parser_grads2[{{}, t, {1, opt.rnn_size_parser}}])
	  local dlst = parser_fwd_clones[t+source_l]:backward(parser_fwd_inputs2[t], drnn_state)
	  for j = 1, #drnn_state do
	    drnn_state[j]:copy(dlst[j+1])
	  end
	  word_vecs2_grads[{{}, t}]:add(dlst[1])
	end
	-- backprop through bwd parser rnn
	local drnn_state = reset_state(init_parser, batch_l)
	for t = 1, target_l do
	  drnn_state[#drnn_state]:add(
	    parser_grads2[{{}, t, {opt.rnn_size_parser+1, 2*opt.rnn_size_parser}}])
	  local dlst = parser_bwd_clones[t+source_l]:backward(parser_bwd_inputs2[t], drnn_state)
	  for j = 1, #drnn_state do
	    drnn_state[j]:copy(dlst[j+1])
	  end
	  word_vecs2_grads[{{}, t}]:add(dlst[1])
	end
	------ backprop through source word vectors ------
	if opt.proj == 0 then
	  word_vecs_enc2:backward(target, word_vecs2_grads)
	end
	
      else
	local dl_dinput1, dl_dinput2 = table.unpack(sent_encoder:backward(pred_input, dl_dp))    
	word_vecs_enc1:backward(source, dl_dinput1)
	word_vecs_enc2:backward(target, dl_dinput2)
      end
      
      if opt.fix_word_vecs == 1 then
	word_vecs_enc1.gradWeight:zero()
	word_vecs_enc2.gradWeight:zero()	   
      end
      
      -- word vec layer and parser_graph layers are shared
      grad_params[1]:add(grad_params[2])
      grad_params[2]:zero()
      if opt.attn ~= 'none' then
	grad_params[6]:add(grad_params[7])
	grad_params[7]:zero()
      end

      if opt.share_params == 1 then
	if opt.proj == 1 then
	  entail_layers.proj1.gradWeight:add(entail_layers.proj2.gradWeight)
	  entail_layers.proj2.gradWeight:zero()
	end	    
	for k = 2, 5, 3 do	       
	  entail_layers.f1.modules[k].gradWeight:add(entail_layers.f2.modules[k].gradWeight)
	  entail_layers.f1.modules[k].gradBias:add(entail_layers.f2.modules[k].gradBias)
	  entail_layers.g1.modules[k].gradWeight:add(entail_layers.g2.modules[k].gradWeight)
	  entail_layers.g1.modules[k].gradBias:add(entail_layers.g2.modules[k].gradBias)
	  entail_layers.f2.modules[k].gradWeight:zero()
	  entail_layers.f2.modules[k].gradBias:zero()
	  entail_layers.g2.modules[k].gradWeight:zero()
	  entail_layers.g2.modules[k].gradBias:zero()
	end
	
      end	 
      
      local grad_norm = 0
      for i = 1, #grad_params do
	grad_norm = grad_norm + grad_params[i]:norm()^2
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
	if j ~= 2 and j ~= 6 then
	  param_norm = param_norm + params[j]:norm()^2
	end	    
      end	 
      param_norm = param_norm^0.5

      params[2]:copy(params[1])
      if opt.attn ~= 'none' then
	params[7]:copy(params[6])
      end

      if opt.share_params == 1 then
	if opt.proj == 1 then
	  entail_layers.proj2.weight:copy(entail_layers.proj1.weight)
	end	    
	for k = 2, 5, 3 do	 
	  entail_layers.f2.modules[k].weight:copy(entail_layers.f1.modules[k].weight)
	  entail_layers.f2.modules[k].bias:copy(entail_layers.f1.modules[k].bias)
	  entail_layers.g2.modules[k].weight:copy(entail_layers.g1.modules[k].weight)
	  entail_layers.g2.modules[k].bias:copy(entail_layers.g1.modules[k].bias)
	end      
      end
      
      -- Bookkeeping
      num_words_target = num_words_target + batch_l*target_l
      num_words_source = num_words_source + batch_l*source_l
      train_loss = train_loss + loss
      train_sents = train_sents + batch_l
      local time_taken = timer:time().real - start_time
      if i % opt.print_every == 0 then
	local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
				    epoch, i, data:size(), batch_l, opt.learning_rate)
	stats = stats .. string.format('NLL: %.4f, Acc: %.4f, |Param|: %.2f, |GParam|: %.2f, ',
				       train_loss/train_sents, train_num_correct/train_sents,
				       param_norm, grad_norm)
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
    return train_loss, train_sents, train_num_correct
  end
  -- eval(valid_data)
  -- eval(test_data)
  local best_val_perf = 0
  local test_perf = 0
  for epoch = 1, opt.epochs do
    local total_loss, total_sents, total_correct = train_batch(train_data, epoch)
    local train_score = total_correct/total_sents
    print('Train', train_score)
    opt.train_perf[#opt.train_perf + 1] = train_score
    local score = eval(valid_data)
    local savefile = string.format('%s.t7', opt.savefile)            
    if score > best_val_perf then
      best_val_perf = score
      test_perf = eval(test_data)
      print('saving checkpoint to ' .. savefile)
      torch.save(savefile, {layers, opt, layer_etas, optStates})	 	 
    end
    opt.val_perf[#opt.val_perf + 1] = score
    print(opt.train_perf)
    print(opt.val_perf)
  end
  print("Best Val", best_val_perf)
  print("Test", test_perf)   
  -- save final model
  local savefile = string.format('%s_final.t7', opt.savefile)
  print('saving final model to ' .. savefile)
  for i = 1, #layers do
    layers[i]:double()
  end   
  torch.save(savefile, {layers, opt, layer_etas, optStates})
end

function eval(data)
  sent_encoder:evaluate()
  local nll = 0
  local num_sents = 0
  local num_correct = 0
  for i = 1, data:size() do
    local d = data[i]
    local target, source, batch_l, target_l, source_l, label = table.unpack(d)
    local sent1_context = sent1_context_proto[{{1, batch_l}}]
    local sent2_context = sent2_context_proto[{{1, batch_l}}]      
    local word_vecs1 = word_vecs_enc1:forward(source) 	 
    local word_vecs2 = word_vecs_enc2:forward(target)
    if opt.attn ~= 'none' then
      set_size_encoder(batch_l, source_l, target_l,
		       opt.word_vec_size, opt.hidden_size, entail_layers)
      set_size_parser(batch_l, source_l, opt.rnn_size_parser*2, parser_layers1)
      set_size_parser(batch_l, target_l, opt.rnn_size_parser*2, parser_layers2)
      
      -- resize the various temporary tensors that are going to hold contexts/grads
      local parser_context1 = parser_context1_proto[{{1, batch_l}, {1, source_l}}]
      local parser_context2 = parser_context2_proto[{{1, batch_l}, {1, target_l}}]

      ------ fwd prop for parser brnn for sent 1------
      -- fwd direction
      local rnn_state_parser_fwd1 = reset_state(init_parser, batch_l, 0)
      local parser_fwd_inputs1 = {}	 
      for t = 1, source_l do	    
	parser_fwd_clones[t]:evaluate()
	parser_fwd_inputs1[t] = {word_vecs1[{{}, t}], table.unpack(rnn_state_parser_fwd1[t-1])}
	local out = parser_fwd_clones[t]:forward(parser_fwd_inputs1[t])
	rnn_state_parser_fwd1[t] = out
	parser_context1[{{}, t, {1, opt.rnn_size_parser}}]:copy(out[#out])
      end
      -- bwd direction
      local rnn_state_parser_bwd1 = reset_state(init_parser, batch_l, source_l+1)
      local parser_bwd_inputs1 = {}	 
      for t = source_l, 1, -1 do
	parser_bwd_clones[t]:evaluate()
	parser_bwd_inputs1[t] = {word_vecs1[{{}, t}], table.unpack(rnn_state_parser_bwd1[t+1])}
	local out = parser_bwd_clones[t]:forward(parser_bwd_inputs1[t])
	rnn_state_parser_bwd1[t] = out
	parser_context1[{{}, t, {opt.rnn_size_parser+1, opt.rnn_size_parser*2}}]:copy(out[#out])
      end

      ------ fwd prop for parser brnn for sent 2------
      -- fwd direction
      local rnn_state_parser_fwd2 = reset_state(init_parser, batch_l, 0)
      local parser_fwd_inputs2 = {}	 
      for t = 1, target_l do	    
	parser_fwd_clones[t+source_l]:training()
	parser_fwd_inputs2[t] = {word_vecs2[{{}, t}], table.unpack(rnn_state_parser_fwd2[t-1])} 
	local out = parser_fwd_clones[t+source_l]:forward(parser_fwd_inputs2[t])
	rnn_state_parser_fwd2[t] = out
	parser_context2[{{}, t, {1, opt.rnn_size_parser}}]:copy(out[#out])
      end
      -- bwd direction
      local rnn_state_parser_bwd2 = reset_state(init_parser, batch_l, target_l+1)
      local parser_bwd_inputs2 = {}	 
      for t = target_l, 1, -1 do
	parser_bwd_clones[t+source_l]:training()
	parser_bwd_inputs2[t] = {word_vecs2[{{}, t}], table.unpack(rnn_state_parser_bwd2[t+1])}
	local out = parser_bwd_clones[t+source_l]:forward(parser_bwd_inputs2[t])
	rnn_state_parser_bwd2[t] = out
	parser_context2[{{}, t, {opt.rnn_size_parser+1, opt.rnn_size_parser*2}}]:copy(out[#out])
      end	 
      parsed_context1 = parser_graph1:forward(parser_context1:contiguous())
      parsed_context2 = parser_graph2:forward(parser_context2:contiguous())
      pred_input = {word_vecs1, word_vecs2, parsed_context1, parsed_context2}
    else
      set_size_encoder(batch_l, source_l, target_l,
		       opt.word_vec_size, opt.hidden_size, entail_layers)	 	 
      pred_input = {word_vecs1, word_vecs2}
    end      
    local pred_label = sent_encoder:forward(pred_input)
    local loss = disc_criterion:forward(pred_label, label)
    local _, pred_argmax = pred_label:max(2)
    num_correct = num_correct + pred_argmax:double():view(batch_l):eq(label:double()):sum()
    num_sents = num_sents + batch_l
    nll = nll + loss
  end
  local acc = num_correct/num_sents
  print("Acc", acc)
  print("NLL", nll / num_sents)
  collectgarbage()
  return acc
end


function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vecs_enc2' then
      word_vecs_dec = layer
    elseif layer.name == 'parser' then
      parser = layer
    end      
  end
end

function main() 
  -- parse input params
  opt = cmd:parse(arg)
  if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    if opt.cudnn == 1 then
      print('loading cudnn...')
      require 'cudnn'
    end      
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)      
  end
  
  -- Create the data loader class.
  print('loading data...')

  train_data = data.new(opt, opt.data_file)   
  valid_data = data.new(opt, opt.val_data_file)
  test_data = data.new(opt, opt.test_data_file)
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
  
  -- Build model
  word_vecs_enc1 = nn.LookupTable(valid_data.source_size, opt.word_vec_size)
  word_vecs_enc2 = nn.LookupTable(valid_data.target_size, opt.word_vec_size)
  if opt.attn ~= 'none' then
    parser_fwd = make_lstm(valid_data, opt.rnn_size_parser, opt.word_vec_size,
			   opt.num_layers_parser, opt, 'enc')
    parser_bwd = make_lstm(valid_data, opt.rnn_size_parser, opt.word_vec_size,
			   opt.num_layers_parser, opt, 'enc')
    parser_graph1 = make_parser(opt.rnn_size_parser*2)
    parser_graph2 = make_parser(opt.rnn_size_parser*2)
    sent_encoder = make_sent_encoder(opt.word_vec_size, opt.hidden_size,
				     valid_data.label_size, opt.dropout)
  else	 
    sent_encoder = make_sent_encoder(opt.word_vec_size, opt.hidden_size,
				     valid_data.label_size, opt.dropout)	 
  end      

  disc_criterion = nn.ClassNLLCriterion()
  disc_criterion.sizeAverage = false

  
  if opt.attn ~= 'none' then
    layers = {word_vecs_enc1, word_vecs_enc2, sent_encoder,
	      parser_fwd, parser_bwd,
	      parser_graph1, parser_graph2}
  else
    layers = {word_vecs_enc1, word_vecs_enc2, sent_encoder}
  end

  layer_etas = {}
  optStates = {}   
  for i = 1, #layers do
    layer_etas[i] = opt.learning_rate -- can have layer-specific lr, if desired
    optStates[i] = {}
  end
  
  if opt.gpuid >= 0 then
    for i = 1, #layers do	 
      layers[i]:cuda()
    end
    disc_criterion:cuda()
  end

  -- these layers will be manipulated during training
  if opt.attn ~= 'none' then
    parser_layers1 = {}
    parser_layers2 = {}
    parser_graph1:apply(get_parser_layer1)
    parser_graph2:apply(get_parser_layer2)
  end
  entail_layers = {}   
  sent_encoder:apply(get_entail_layer)
  if opt.attn ~= 'none' then
    if opt.cuda_mod == 1 then
      require 'cuda-mod'
      parser_layers1.dep_parser.cuda_mod = 1
      parser_layers2.dep_parser.cuda_mod = 1
    else
      if opt.attn == 'struct' then
	parser_layers1.dep_parser:double()
	parser_layers2.dep_parser:double()
      end      
    end      
  end
  train(train_data, valid_data)
end

main()
