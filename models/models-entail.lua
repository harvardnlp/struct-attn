require 'models/EisnerCRF.lua'

function nn.Module:reuseMem()
   self.reuse = true
   return self
end

function nn.Module:setReuse()
   if self.reuse then
      self.gradInput = self.output
   end
end

function make_lstm(data, rnn_size, input_size, n, opt, model)
   local name = '_' .. model
   local dropout = opt.dropout or 0
  -- there will be 2*n+3 inputs
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
   for L = 1,n do
      table.insert(inputs, nn.Identity()()) -- prev_c[L]
      table.insert(inputs, nn.Identity()()) -- prev_h[L]
   end

   local x, input_size_L
   local outputs = {}
  for L = 1,n do
     -- c,h from previous timesteps
    local prev_c = inputs[L*2]    
    local prev_h = inputs[L*2+1]
    -- the input to this layer
    if L == 1 then
       local word_vecs
       if model == 'enc' then
	 x = inputs[1]
       else
	  local word_vecs = nn.LookupTable(data.target_size, input_size)
	  word_vecs.name = 'word_vecs' .. name
	  x = word_vecs(inputs[1]) -- batch_size x word_vec_size	  
       end	  
       input_size_L = input_size
    else
       x = outputs[(L-1)*2]
       input_size_L = rnn_size
       if dropout > 0 then
	  x = nn.Dropout(dropout, nil, false)(x)
       end       
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):reuseMem()(x)
    local h2h_layer = nn.Linear(rnn_size, 4*rnn_size, false)
    local h2h = h2h_layer(prev_h)
    h2h_layer.name = 'h2h'
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
  return nn.gModule(inputs, outputs)
end

function make_parser(rnn_size)
   --placeholders (these will change during training)   
   local sent_l = 5 
   local batch_l = 1   
   local input = nn.Identity()() -- batch_l x sent_l x rnn_size
   local parser_view = nn.View(batch_l*sent_l, rnn_size)
   parser_view.name = 'parser_view'   
   local score_input = parser_view(input) -- move to batch_l*sent_l x rnn_size
   local parent = nn.Linear(rnn_size, rnn_size)(score_input)
   local child = nn.Linear(rnn_size, rnn_size, false)(score_input)
   local parent_view = nn.View(batch_l, sent_l, rnn_size)
   local child_view = nn.View(batch_l, sent_l, rnn_size)   
   parent_view.name = 'parent_view'
   child_view.name = 'child_view'
   parent = parent_view(parent) -- get back to batch_l x sent_l x rnn_size
   child = child_view(child)
   local parent_rep = nn.Replicate(sent_l, 2, 4)
   local child_rep = nn.Replicate(sent_l, 3, 4)
   parent_rep.name = 'parent_rep'
   child_rep.name = 'child_rep'
   parent = parent_rep(parent)
   child = child_rep(child)
   local scores = nn.Tanh()(nn.CAddTable()({parent, child})) -- batch_l x sent_l x sent_l x rnn_size
   local score_view = nn.View(batch_l*sent_l*sent_l, rnn_size)
   score_view.name = 'score_view'
   scores = nn.Linear(rnn_size, 1, false)(score_view(scores)) -- batch_l*sent_l*sent_l x 1
   local score_unview = nn.View(batch_l, sent_l, sent_l)
   score_unview.name = 'score_unview'
   scores = score_unview(scores)
   local dep_parser = nn.EisnerCRF()
   dep_parser.name = 'dep_parser'
   local output
   if opt.gpuid >= 0 then
     output = nn.gpu()(dep_parser(nn.cpu()(nn.Tanh()(scores))))
   else
     output = dep_parser(nn.Tanh()(scores))
   end   
   return nn.gModule({input}, {output})
end

function get_parser_layer1(layer)
   if layer.name ~= nil then
      parser_layers1[layer.name] = layer
   end   
end

function get_parser_layer2(layer)
   if layer.name ~= nil then
      parser_layers2[layer.name] = layer
   end   
end

function get_entail_layer(layer)
   if layer.name ~= nil then
      entail_layers[layer.name] = layer
   end   
end

function set_size_parser(batch_l, sent_l, rnn_size, t)
   t.parser_view.size[1] = batch_l*sent_l
   t.parser_view.numElements = batch_l*sent_l*rnn_size
   
   t.parent_view.size[1] = batch_l
   t.parent_view.size[2] = sent_l
   t.parent_view.numElements = batch_l*sent_l*rnn_size
   
   t.child_view.size[1] = batch_l
   t.child_view.size[2] = sent_l
   t.child_view.numElements = batch_l*sent_l*rnn_size

   if t.parent_rep ~= nil then
     t.parent_rep.nfeatures = sent_l
     t.child_rep.nfeatures = sent_l
     if t.input2_rep ~= nil then
       t.input2_rep.nfeatures = sent_l
     end   
     
     t.score_view.size[1] = batch_l*sent_l*sent_l
     t.score_view.numElements = batch_l*sent_l*sent_l*rnn_size
     
     t.score_unview.size[1] = batch_l
     t.score_unview.size[2] = sent_l
     t.score_unview.size[3] = sent_l
     t.score_unview.numElements = batch_l*sent_l*sent_l
   end
   if t.softmax_view ~= nil then
     t.softmax_view.size[1] = batch_l*sent_l
     t.softmax_view.size[2] = sent_l
     t.softmax_view.numElements = batch_l*sent_l*sent_l
     t.softmax_unview.size[1] = batch_l
     t.softmax_unview.size[2] = sent_l
     t.softmax_unview.size[3] = sent_l
     t.softmax_unview.numElements = batch_l*sent_l*sent_l     
   end   
end

function set_size_encoder(batch_l, sent_l1, sent_l2, input_size, hidden_size, t)
   local size
   if opt.proj == 1 then
      size = hidden_size
   else
      size = input_size
   end
   if opt.parser == 1 then
      size = size * (opt.use_parent + opt.use_children + 1)
   end
   
   if opt.proj == 1 then
      t.input1_proj_view.size[1] = batch_l*sent_l1
      t.input1_proj_view.numElements = batch_l*sent_l1*input_size
      t.input2_proj_view.size[1] = batch_l*sent_l2
      t.input2_proj_view.numElements = batch_l*sent_l2*input_size

      t.input1_proj_unview.size[1] = batch_l
      t.input1_proj_unview.size[2] = sent_l1
      t.input1_proj_unview.numElements = batch_l*sent_l1*hidden_size
      t.input2_proj_unview.size[1] = batch_l
      t.input2_proj_unview.size[2] = sent_l2
      t.input2_proj_unview.numElements = batch_l*sent_l2*hidden_size
   end

   t.input1_view.size[1] = batch_l*sent_l1
   t.input1_view.numElements = batch_l*sent_l1*size   
   t.input1_unview.size[1] = batch_l
   t.input1_unview.size[2] = sent_l1
   t.input1_unview.numElements = batch_l*sent_l1*hidden_size
   
   t.input2_view.size[1] = batch_l*sent_l2
   t.input2_view.numElements = batch_l*sent_l2*size
   t.input2_unview.size[1] = batch_l
   t.input2_unview.size[2] = sent_l2
   t.input2_unview.numElements = batch_l*sent_l2*hidden_size     
   
   t.scores1_view.size[1] = batch_l*sent_l1
   t.scores1_view.size[2] = sent_l2
   t.scores1_view.numElements = batch_l*sent_l1*sent_l2
   t.scores2_view.size[1] = batch_l*sent_l2
   t.scores2_view.size[2] = sent_l1
   t.scores2_view.numElements = batch_l*sent_l1*sent_l2

   t.scores1_unview.size[1] = batch_l
   t.scores1_unview.size[2] = sent_l1
   t.scores1_unview.size[3] = sent_l2
   t.scores1_unview.numElements = batch_l*sent_l1*sent_l2
   t.scores2_unview.size[1] = batch_l
   t.scores2_unview.size[2] = sent_l2 
   t.scores2_unview.size[3] = sent_l1  
   t.scores2_unview.numElements = batch_l*sent_l1*sent_l2

   if opt.extra_feat == 1 then
     t.input1_combined_view.size[1] = batch_l*sent_l1
     t.input1_combined_view.numElements = batch_l*sent_l1*4*size
     t.input2_combined_view.size[1] = batch_l*sent_l2
     t.input2_combined_view.numElements = batch_l*sent_l2*4*size
   else
     t.input1_combined_view.size[1] = batch_l*sent_l1
     t.input1_combined_view.numElements = batch_l*sent_l1*2*size
     t.input2_combined_view.size[1] = batch_l*sent_l2
     t.input2_combined_view.numElements = batch_l*sent_l2*2*size     
   end

   t.input1_combined_unview.size[1] = batch_l
   t.input1_combined_unview.size[2] = sent_l1
   t.input1_combined_unview.numElements = batch_l*sent_l1*hidden_size
   t.input2_combined_unview.size[1] = batch_l
   t.input2_combined_unview.size[2] = sent_l2
   t.input2_combined_unview.numElements = batch_l*sent_l2*hidden_size
end

function make_sent_encoder(input_size, hidden_size, num_labels, dropout)
   local sent_l1 = 5 -- sent_l1, sent_l2, and batch_l are default values that will change 
   local sent_l2 = 10
   local batch_l = 1
   local inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local input1 = inputs[1] -- batch_l x sent_l1 x input_size
   local input2 = inputs[2] --batch_l x sent_l2 x input_size
   
   local input1_proj, input2_proj, size
   if opt.proj == 1 then
      local proj1 = nn.Linear(input_size, hidden_size, false)
      local proj2 = nn.Linear(input_size, hidden_size, false)
      proj1.name = 'proj1'
      proj2.name = 'proj2'
      local input1_proj_view = nn.View(batch_l*sent_l1, input_size)
      local input2_proj_view = nn.View(batch_l*sent_l2, input_size)
      local input1_proj_unview = nn.View(batch_l, sent_l1, hidden_size)
      local input2_proj_unview = nn.View(batch_l, sent_l2, hidden_size)   
      input1_proj_view.name = 'input1_proj_view'
      input2_proj_view.name = 'input2_proj_view'
      input1_proj_unview.name = 'input1_proj_unview'
      input2_proj_unview.name = 'input2_proj_unview'
      input1_proj = input1_proj_unview(proj1(input1_proj_view(input1))) 
      input2_proj = input2_proj_unview(proj2(input2_proj_view(input2)))      
      size = hidden_size      
   else
      input1_proj = input1
      input2_proj = input2
      size = input_size
   end
   
   if opt.parser == 1 then   
      table.insert(inputs, nn.Identity()()) -- batch_l x sent_l1 x sent_l1
      table.insert(inputs, nn.Identity()()) -- batch_l x sent_l2 x sent_l2
      local input1_tmp = input1_proj
      local input2_tmp = input2_proj
      if opt.use_parent == 1 then
	 local input1_parent = nn.MM()({nn.Transpose({2,3})(inputs[3]), input1_tmp})
	 local input2_parent = nn.MM()({nn.Transpose({2,3})(inputs[4]), input2_tmp})
	 input1_proj = nn.JoinTable(3)({input1_proj, input1_parent})
	 input2_proj = nn.JoinTable(3)({input2_proj, input2_parent})	 
      end
      if opt.use_children == 1 then
	 local input1_children = nn.MM()({inputs[3], input1_tmp})
	 local input2_children = nn.MM()({inputs[4], input2_tmp})
	 input1_proj = nn.JoinTable(3)({input1_proj, input1_children})
	 input2_proj = nn.JoinTable(3)({input2_proj, input2_children})
      end      
      size = size*(1 + opt.use_parent + opt.use_children)
   end

   local f1 = nn.Sequential()
   f1:add(nn.Dropout(dropout))      
   f1:add(nn.Linear(size, hidden_size))
   f1:add(nn.ReLU())
   f1:add(nn.Dropout(dropout))
   f1:add(nn.Linear(hidden_size, hidden_size))
   f1:add(nn.ReLU())
   f1.name = 'f1'
   local f2 = nn.Sequential()
   f2:add(nn.Dropout(dropout))   
   f2:add(nn.Linear(size, hidden_size))
   f2:add(nn.ReLU())
   f2:add(nn.Dropout(dropout))
   f2:add(nn.Linear(hidden_size, hidden_size))
   f2:add(nn.ReLU())
   f2.name = 'f2'
   local input1_view = nn.View(batch_l*sent_l1, size)
   local input2_view = nn.View(batch_l*sent_l2, size)
   local input1_unview = nn.View(batch_l, sent_l1, hidden_size)
   local input2_unview = nn.View(batch_l, sent_l2, hidden_size)   
   input1_view.name = 'input1_view'
   input2_view.name = 'input2_view'
   input1_unview.name = 'input1_unview'
   input2_unview.name = 'input2_unview'

   local input1_hidden = input1_unview(f1(input1_view(input1_proj)))
   local input2_hidden = input2_unview(f2(input2_view(input2_proj)))
   local scores1 = nn.MM()({input1_hidden,
			    nn.Transpose({2,3})(input2_hidden)}) -- batch_l x sent_l1 x sent_l2
   local scores2 = nn.Transpose({2,3})(scores1) -- batch_l x sent_l2 x sent_l1

   local scores1_view = nn.View(batch_l*sent_l1, sent_l2)
   local scores2_view = nn.View(batch_l*sent_l2, sent_l1)
   local scores1_unview = nn.View(batch_l, sent_l1, sent_l2)
   local scores2_unview = nn.View(batch_l, sent_l2, sent_l1)
   scores1_view.name = 'scores1_view'
   scores2_view.name = 'scores2_view'
   scores1_unview.name = 'scores1_unview'
   scores2_unview.name = 'scores2_unview'
  
   local prob1 = scores1_unview(nn.SoftMax()(scores1_view(scores1))) 
   local prob2 = scores2_unview(nn.SoftMax()(scores2_view(scores2)))
  
   local input2_soft = nn.MM()({prob1, input2_proj}) -- batch_l x sent_l1 x input_size
   local input1_soft = nn.MM()({prob2, input1_proj}) -- batch_l x sent_l2 x input_size

   local new_size, input1_combined, input2_combined
   if opt.extra_feat == 1 then
     local input1_minus = nn.CAddTable()({input1_proj, nn.MulConstant(-1)(input2_soft)})
     local input2_minus = nn.CAddTable()({input2_proj, nn.MulConstant(-1)(input1_soft)})
     local input1_prod = nn.CMulTable()({input1_proj, input2_soft})
     local input2_prod = nn.CMulTable()({input2_proj, input1_soft})
     input1_combined = nn.JoinTable(3)({input1_proj, input2_soft, input1_minus, input1_prod}) -- batch_l x sent_l1 x input_size*2
     input2_combined = nn.JoinTable(3)({input2_proj, input1_soft, input2_minus, input2_prod}) -- batch_l x sent_l2 x input_size*2
     new_size = size*4
   else
     input1_combined = nn.JoinTable(3)({input1_proj ,input2_soft}) -- batch_l x sent_l1 x input_size*2
     input2_combined = nn.JoinTable(3)({input2_proj,input1_soft}) -- batch_l x sent_l2 x input_size*2
     new_size = size*2
   end
   local input1_combined_view = nn.View(batch_l*sent_l1, new_size)
   local input2_combined_view = nn.View(batch_l*sent_l2, new_size)
   local input1_combined_unview = nn.View(batch_l, sent_l1, hidden_size)
   local input2_combined_unview = nn.View(batch_l, sent_l2, hidden_size)
   input1_combined_view.name = 'input1_combined_view'
   input2_combined_view.name = 'input2_combined_view'
   input1_combined_unview.name = 'input1_combined_unview'
   input2_combined_unview.name = 'input2_combined_unview'

   local g1 = nn.Sequential()
   g1:add(nn.Dropout(dropout))   
   g1:add(nn.Linear(new_size, hidden_size))
   g1:add(nn.ReLU())
   g1:add(nn.Dropout(dropout))      
   g1:add(nn.Linear(hidden_size, hidden_size))
   g1:add(nn.ReLU())
   g1.name = 'g1'
   local g2 = nn.Sequential()
   g2:add(nn.Dropout(dropout))
   g2:add(nn.Linear(new_size, hidden_size))
   g2:add(nn.ReLU())
   g2:add(nn.Dropout(dropout))         
   g2:add(nn.Linear(hidden_size, hidden_size))
   g2:add(nn.ReLU())
   g2.name = 'g2'
   local input1_output = input1_combined_unview(g1(input1_combined_view(input1_combined)))
   local input2_output = input2_combined_unview(g2(input2_combined_view(input2_combined)))
   input1_output = nn.Sum(2)(input1_output) -- batch_l x hidden_size
   input2_output = nn.Sum(2)(input2_output) -- batch_l x hidden_size     
   new_size = hidden_size*2
   
   local join_layer = nn.JoinTable(2)
   local input12_combined = join_layer({input1_output, input2_output})
   join_layer.name = 'join'
   local out_layer = nn.Sequential()
   out_layer:add(nn.Dropout(dropout))
   out_layer:add(nn.Linear(new_size, hidden_size))
   out_layer:add(nn.ReLU())
   out_layer:add(nn.Dropout(dropout))
   out_layer:add(nn.Linear(hidden_size, hidden_size))
   out_layer:add(nn.ReLU())
   out_layer:add(nn.Linear(hidden_size, num_labels))
   out_layer:add(nn.LogSoftMax())
   out_layer.name = 'out_layer'
   local out = out_layer(input12_combined)
   return nn.gModule(inputs, {out})
end

local GPU, _ = torch.class('nn.gpu', 'nn.Module')

function GPU:__init()
   nn.Module.__init(self)
   self.output = torch.CudaTensor()
   self.gradInput = torch.DoubleTensor()
end

function GPU:updateOutput(input)
   self.output = input:cuda()
   return self.output
end

function GPU:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput:double()
   return self.gradInput
end

local CPU, _ = torch.class('nn.cpu', 'nn.Module')

function CPU:__init()
   nn.Module.__init(self)
   self.output = torch.DoubleTensor()
   self.gradInput = torch.CudaTensor()
end

function CPU:updateOutput(input)
   self.output = input:double()
   return self.output
end

function CPU:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput:cuda()
   return self.gradInput
end

