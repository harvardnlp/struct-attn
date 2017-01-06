function nn.Module:reuseMem()
   self.reuse = true
   return self
end

function nn.Module:setReuse()
   if self.reuse then
      self.gradInput = self.output
   end
end

function make_lstm(data, opt, model, use_chars)
   assert(model == 'enc' or model == 'dec')
   local name = '_' .. model
   local dropout = opt.dropout or 0
   local n = opt.num_layers
   local rnn_size = opt.rnn_size
   local input_size
   if use_chars == 0 then
      input_size = opt.word_vec_size
   else
      input_size = opt.num_kernels
   end   
   local offset = 0
  -- there will be 2*n+3 inputs
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
   if model == 'dec' then
      table.insert(inputs, nn.Identity()()) -- all context (batch_size x source_l x rnn_size)
      offset = offset + 1
      if opt.input_feed == 1 then
	 table.insert(inputs, nn.Identity()()) -- prev context_attn (batch_size x rnn_size)
	 offset = offset + 1
      end
   end
   for L = 1,n do
      table.insert(inputs, nn.Identity()()) -- prev_c[L]
      table.insert(inputs, nn.Identity()()) -- prev_h[L]
   end

   local x, input_size_L
   local outputs = {}
  for L = 1,n do
     -- c,h from previous timesteps
    local prev_c = inputs[L*2+offset]    
    local prev_h = inputs[L*2+1+offset]
    -- the input to this layer
    if L == 1 then
       if use_chars == 0 then
	  local word_vecs
	  if model == 'enc' then
	     word_vecs = nn.LookupTable(data.source_size, input_size)
	  else
	     word_vecs = nn.LookupTable(data.target_size, input_size)
	  end	  
	  word_vecs.name = 'word_vecs' .. name
	  x = word_vecs(inputs[1]) -- batch_size x word_vec_size
       else
	  local char_vecs = nn.LookupTable(data.char_size, opt.char_vec_size)
	  char_vecs.name = 'word_vecs' .. name
	  local charcnn = make_cnn(opt.char_vec_size,  opt.kernel_width, opt.num_kernels)
	  charcnn.name = 'charcnn' .. name
	  x = charcnn(char_vecs(inputs[1]))
	  if opt.num_highway_layers > 0 then
	     local mlp = make_highway(input_size, opt.num_highway_layers)
	     mlp.name = 'mlp' .. name
	     x = mlp(x)
	  end	  
       end
       input_size_L = input_size
       if model == 'dec' then
	  if opt.input_feed == 1 then
	     x = nn.JoinTable(2)({x, inputs[1+offset]}) -- batch_size x (word_vec_size + rnn_size)
	     input_size_L = input_size + rnn_size
	  end	  
       end
    else
       x = outputs[(L-1)*2]
       if opt.res_net == 1 and L > 2 then
	  x = nn.CAddTable()({x, outputs[(L-2)*2]})       
       end       
       input_size_L = rnn_size
       if opt.multi_attn == L and model == 'dec' then
	  local multi_attn = make_decoder_attn(data, opt, 1)
	  multi_attn.name = 'multi_attn' .. L
	  x = multi_attn({x, inputs[2]})
       end
       if dropout > 0 then
	  x = nn.Dropout(dropout, nil, false)(x)
       end       
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):reuseMem()(x)
    i2h.data.module.name = 'i2h'
    local h2h = nn.LinearNoBias(rnn_size, 4 * rnn_size):reuseMem()(prev_h)
    h2h.data.module.name = 'h2h'
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():reuseMem()(n1)
    local forget_gate = nn.Sigmoid():reuseMem()(n2)
    local out_gate = nn.Sigmoid():reuseMem()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():reuseMem()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh():reuseMem()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  if model == 'dec' then
     local top_h = outputs[#outputs]
     local decoder_out
     if opt.attn == 1 then
        local decoder_attn
        if opt.attn_type == 'vanilla' then
	    decoder_attn = make_decoder_attn(data, opt)
        elseif opt.attn_type == 'crf' then
            decoder_attn = make_decoder_crf_attn(data, opt)
        elseif opt.attn_type == 'vanilla2' then
             decoder_attn = make_decoder_attn2(data, opt)
        elseif opt.attn_type == 'crf2' then
             decoder_attn = make_decoder_crf_attn2(data, opt)
        elseif opt.attn_type == 'crf3' then
             decoder_attn = make_decoder_crf_attn3(data, opt)
        elseif opt.attn_type == 'crf4' then
             decoder_attn = make_decoder_crf_attn4(data, opt)
        end
	decoder_attn.name = 'decoder_attn'
	decoder_out = decoder_attn({top_h, inputs[2]})
     else
	decoder_out = nn.JoinTable(2)({top_h, inputs[2]})
        dec_linear = nn.Linear(opt.rnn_size*2, opt.rnn_size, false) 
        dec_linear.name = 'dec_linear'
	--decoder_out = nn.Tanh()(nn.LinearNoBias(opt.rnn_size*2, opt.rnn_size)(decoder_out))
	decoder_out = nn.Tanh()(dec_linear(decoder_out))
     end
     if dropout > 0 then
	decoder_out = nn.Dropout(dropout, nil, false)(decoder_out)
     end     
     table.insert(outputs, decoder_out)
  end
  return nn.gModule(inputs, outputs)
end

function make_decoder_attn(data, opt, simple)
   -- 2D tensor target_t (batch_l x rnn_size) and
   -- 3D tensor for context (batch_l x source_l x rnn_size)

   local inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local target_t = nn.LinearNoBias(opt.rnn_size, opt.rnn_size)(inputs[1])
   local context = inputs[2]
   simple = simple or 0
   -- get attention

   local attn = nn.MM()({context, nn.Replicate(1,3)(target_t)}) -- batch_l x source_l x 1
   attn = nn.Sum(3)(attn)
   local softmax_attn = nn.SoftMax()
   softmax_attn.name = 'softmax_attn'
   attn = softmax_attn(attn)
   attn = nn.Replicate(1,2)(attn) -- batch_l x  1 x source_l
   
   -- apply attention to context
   local context_combined = nn.MM()({attn, context}) -- batch_l x 1 x rnn_size
   context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
   local context_output
   if simple == 0 then
      context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
      context_output = nn.Tanh()(nn.LinearNoBias(opt.rnn_size*2,
						 opt.rnn_size)(context_combined))
   else
      context_output = nn.CAddTable()({context_combined,inputs[1]})
   end   
   return nn.gModule(inputs, {context_output})   
end

function make_decoder_crf_attn(data, opt, simple)
   -- 2D tensor target_t (batch_l x rnn_size) and
   -- 3D tensor for context (batch_l x source_l x rnn_size)

   local inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local target_t = nn.LinearNoBias(opt.rnn_size, opt.rnn_size)(inputs[1])
   local context = inputs[2]
   simple = simple or 0
   -- get attention

   local attn = nn.MM()({context, nn.Replicate(1,3)(target_t)}) -- batch_l x source_l x 1
   attn = nn.Sum(3)(attn)
   --local softmax_attn = nn.SoftMax()
   --attn = softmax_attn(attn) 
   attn = nn.Replicate(1,3)(attn) -- batch_l x source_l x 1
   attn = nn.Sigmoid()(attn)
   
   attn_complement = nn.MulConstant(-1)(attn) -- batch_l x source_l x 1
   attn_complement = nn.AddConstant(1)(attn_complement)
   attn = nn.JoinTable(3)({attn, attn_complement}) -- batch_l x source_l x 2
   attn = nn.Log()(attn)
   --attn = nn.Dropout(.5)(attn)
   --attn = nn.Tanh()(attn)
   crf = nn.CRF(2)
   --crf.weight:copy(torch.Tensor({{1, 1}, {1, 0}}))
   attn = crf(attn) -- batch_l x source_l + 1 x 2 x 2 
   attn = nn.Exp()(attn)
   attn = nn.Narrow(2, 2, -1)(attn) -- batch_l x source_l x 2 x 2
   attn = nn.Sum(4)(attn) -- batch_l x source_l x 2
   out = nn.Select(3, 1)
   out.name = 'softmax_attn'
   attn = out(attn)
   --attn = nn.Normalize(2)(attn)
   attn = nn.Replicate(1, 2)(attn) -- batch_l x 1 x source_l 
   
   -- apply attention to context
   local context_combined = nn.MM()({attn, context}) -- batch_l x 1 x rnn_size
   context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
   local context_output
   if simple == 0 then
      context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
      context_output = nn.Tanh()(nn.LinearNoBias(opt.rnn_size*2,
						 opt.rnn_size)(context_combined))
   else
      context_output = nn.CAddTable()({context_combined,inputs[1]})
   end   
   return nn.gModule(inputs, {context_output})   
end

function make_decoder_attn2(data, opt, simple)
   local inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   targ_linear_attn = nn.Linear(opt.rnn_size, opt.rnn_size, false)
   targ_linear_attn.name = 'targ_linear_attn'
   local target_t = targ_linear_attn(inputs[1]) -- batch_l x rnn_size
   local context = inputs[2] -- batch_l x source_l x rnn_size
   local attn = nn.MM()({context, nn.Replicate(1, 3)(target_t)}) -- batch_l x source_l x 1
   attn = nn.Sum(3)(attn) -- batch_l x source_l

   --optional
   --[[
   if opt.attn_tanh then
      attn1 = nn.Tanh(attn1)
      attn2 = nn.Tanh(attn2)
   end
   --]]

   out = nn.SoftMax()
   out.name = 'softmax_attn'
   attn = out(attn)


   attn = nn.Replicate(1, 2)(attn) -- batch_l x 1 x source_l
   context_combined = nn.MM()({attn, context}) -- batch x 1 x rnn_size
   context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
   context_output = nn.CAddTable()({context_combined, inputs[1]})
   return nn.gModule(inputs, {context_output})   
end

function make_decoder_crf_attn2(data, opt, simple)
   local inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local context = inputs[2] -- batch_l x source_l x rnn_size

   targ_linear = nn.Linear(opt.rnn_size, opt.rnn_size*2, false)
   targ_linear.name = 'targ_linear'
   local target_t = targ_linear(inputs[1]) -- batch_l x 2*rnn_size
   local target_t1 = nn.Narrow(2, 1, opt.rnn_size)(target_t) -- batch_l x rnn_size
   local target_t2 = nn.Narrow(2, opt.rnn_size+1, opt.rnn_size)(target_t) -- batch_l x rnn_size

   local attn1 = nn.MM()({context, nn.Replicate(1, 3)(target_t1)})   
   local attn2 = nn.MM()({context, nn.Replicate(1, 3)(target_t2)})
   --attn2 = nn.MulConstant(0)(attn2)

   local attn = nn.JoinTable(3)({attn1, attn2})
   --attn = nn.MulConstant(.1)(attn)
   --attn = nn.Tanh()(attn)
   --attn = nn.Sigmoid()(attn)
   --attn = nn.SoftPlus()(attn)
   --attn = nn.Log()(attn)
   --attn = nn.Sigmoid()(attn)
   --attn = nn.Log()(attn)

   crf = nn.CRF(2)
   attn = crf(attn)
   attn = nn.Exp()(attn)
   attn = nn.Narrow(2, 2, -1)(attn)
   attn = nn.Sum(4)(attn)
   attn = nn.Select(3, 1)(attn)

   out = nn.Normalize(1)
   --out = nn.Identity()
   out.name = 'softmax_attn'
   attn = out(attn)

   attn = nn.Replicate(1, 2)(attn)
   local context_combined = nn.MM()({attn, context})
   context_combined = nn.Sum(2)(context_combined)
   local context_output = nn.CAddTable()({context_combined, inputs[1]})

   return nn.gModule(inputs, {context_output})
end

function make_decoder_crf_attn3(data, opt, simple)
   local inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local target = inputs[1] -- batch_l x rnn_size
   local context = inputs[2] -- batch_l x source_l x rnn_size

   --[[
   local target_t = nn.Replicate(opt.num_tags, 3)(target) -- batch_l x rnn_size x num_tags
   target_t = nn.Bottle(nn.Linear(opt.num_tags, 1)) -- batch_l x rnn_size x 1
   target_t = nn.Sum(3)(target_t) -- batch_l x rnn_size
   --]]

   local target_t = nn.Linear(opt.rnn_size, opt.rnn_size*opt.num_tags, false)(target) -- batch_l x opt.num_tags*rnn_size
   target_t = nn.View(-1, opt.rnn_size, opt.num_tags)(target_t) -- batch_l x opt.rnn_size x opt.num_tags
   local tag_emb = nn.MM()({context, target_t}) -- batch_l x source_l x opt.num_tags 

   tag_emb = nn.Tanh()(tag_emb)
   crf = nn.CRF(opt.num_tags)
   tag_emb = crf(tag_emb) -- batch_l x source_l + 1 x opt.num_tags x opt.num_tags
   tag_emb = nn.Exp()(tag_emb) -- batch_l x source_l x opt.num_tags x opt.num_tags
   tag_emb = nn.Narrow(2, 1, -2)(tag_emb) -- batch_l x source_l x num_tags x num_tags
   tag_emb = nn.Sum(4)(tag_emb) -- batch_l x source_l x num_tags
   --tag_emb = nn.PrintSize('1')(tag_emb)
   tag_emb = nn.Bottle(nn.Linear(opt.num_tags, opt.tag_emb, false))(tag_emb) -- batch_l x source_l x tag_emb

   local context_tagged = nn.JoinTable(3)({context, tag_emb}) -- batch_l x source_l x rnn_size + tag_emb
   context_tagged = nn.Bottle(nn.Linear(opt.rnn_size+opt.tag_emb, opt.rnn_size))(context_tagged) -- batch_l x source_l x rnn_size

   local attn = nn.MM()({context_tagged, nn.Replicate(1, 3)(target)}) -- batch_l x source_l x 1 
   attn = nn.Sum(3)(attn)

   out = nn.Identity()
   out.name = 'softmax_attn'
   attn = out(attn)
   
   attn = nn.Replicate(1, 2)(attn) -- batch_l x 1 x source_l
   context_combined = nn.MM()({attn, context}) -- batch x 1 x rnn_size
   context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
   local context_output = nn.CAddTable()({context_combined, inputs[1]})

   return nn.gModule(inputs, {context_output})
end

function make_decoder_crf_attn4(data, opt, simple)
   local inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local target_t = nn.Linear(opt.rnn_size, opt.rnn_size, false)(inputs[1]) -- batch_l x rnn_size
   local context = inputs[2] -- batch_l x source_l x rnn_size
   local attn = nn.MM()({context, nn.Replicate(1, 3)(target_t)}) -- batch_l x source_l x 1
   attn = nn.Replicate(1, 3)(nn.Sum(3)(attn)) -- batch_l x source_l x 1
   attn_0 = nn.Replicate(1, 3)(nn.MulConstant(0)(attn)) -- batch_l x source_l x 1
   attn_2 = nn.Replicate(1, 3)(nn.MulConstant(0)(attn)) -- batch_l x source_l x 1
   attn = nn.JoinTable(3)({attn_0, attn, attn_2}) -- batch_l x source_l x 3
   crf = nn.CRF_static(3, torch.Tensor({{0, 0, -math.huge}, {-math.huge, 0, 0}, {-math.huge, -math.huge, 0}}))
   --print('weight', crf.weight)
   --local weight = torch.Tensor({{0, 0, -math.huge}, {-math.huge, -math.huge, 0}, {-math.huge, -math.huge, 0}})
   --weight = torch.randn(3, 3)
   --crf.weight:copy(weight)

   --print('weight', crf.weight)
   attn = crf(attn) -- batch_l x source_l + 1 x 3 x 3

   attn = nn.Exp()(attn)
   --attn = nn.Print('Positional Marginals')(attn)
   attn = nn.Sum(4)(attn) -- batch_l x source_l+1 x 3
   attn = nn.Narrow(2, 2, -1)(attn) -- batch_l x source_l x 3
   attn = nn.Select(3, 2)(attn) -- batch_l x source_l
   --attn = nn.Sum(3)(attn) -- batch_l x source_l
   --attn = nn.Normalize(2)(attn)

   out = nn.Identity()
   out.name = 'softmax_attn'
   attn = out(attn)

   attn = nn.Replicate(1, 2)(attn) -- batch_l x 1 x source_l
   --context = nn.PrintSize('2')(context) 
   --attn = nn.PrintSize('1')(attn)
   context_combined = nn.MM()({attn, context}) -- batch x 1 x rnn_size
   context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
   context_output = nn.CAddTable()({context_combined, inputs[1]})
   return nn.gModule(inputs, {context_output})   
end


function make_generator(data, opt)
   local model = nn.Sequential()
   model:add(nn.Linear(opt.rnn_size, data.target_size))
   model:add(nn.LogSoftMax())
   local w = torch.ones(data.target_size)
   w[1] = 0
   criterion = nn.ClassNLLCriterion(w)
   criterion.sizeAverage = false
   return model, criterion
end

-- cnn Unit
function make_cnn(input_size, kernel_width, num_kernels)
   local output
   local input = nn.Identity()() 
   if opt.cudnn == 1 then
      local conv = cudnn.SpatialConvolution(1, num_kernels, input_size,
					    kernel_width, 1, 1, 0)
      local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input))
      output = nn.Sum(3)(nn.Max(3)(nn.Tanh()(conv_layer)))
   else
      local conv = nn.TemporalConvolution(input_size, num_kernels, kernel_width)
      local conv_layer = conv(input)
      output = nn.Max(2)(nn.Tanh()(conv_layer))
   end
   return nn.gModule({input}, {output})
end

function make_highway(input_size, num_layers, output_size, bias, f)
    -- size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- bias = bias for transform gate (default = -2)
    -- f = non-linearity (default = ReLU)
    
    local num_layers = num_layers or 1
    local input_size = input_size
    local output_size = output_size or input_size
    local bias = bias or -2
    local f = f or nn.ReLU()
    local start = nn.Identity()()
    local transform_gate, carry_gate, input, output
    for i = 1, num_layers do
       if i > 1 then
	  input_size = output_size
       else
	  input = start
       end       
       output = f(nn.Linear(input_size, output_size)(input))
       transform_gate = nn.Sigmoid()(nn.AddConstant(bias, true)(
					nn.Linear(input_size, output_size)(input)))
       carry_gate = nn.AddConstant(1, true)(nn.MulConstant(-1)(transform_gate))
       local proj
       if input_size==output_size then
	  proj = nn.Identity()
       else
	  proj = nn.LinearNoBias(input_size, output_size)
       end
       input = nn.CAddTable()({
	                     nn.CMulTable()({transform_gate, output}),
                             nn.CMulTable()({carry_gate, proj(input)})})
    end
    return nn.gModule({start},{input})
end

