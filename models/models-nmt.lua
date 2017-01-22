require 'models/memory.lua'
require 'models/CRF.lua'
require 'models/Markov.lua'
require 'models/Util.lua'

function make_lstm(data, opt, model)
  assert(model == 'enc' or model == 'dec')
  local name = '_' .. model
  local dropout = opt.dropout or 0
  local n = opt.num_layers
  local rnn_size = opt.rnn_size
  local RnnD={opt.rnn_size,opt.rnn_size}
  local input_size = opt.word_vec_size
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
    local nameL=model..'_L'..L..'_'
    -- c,h from previous timesteps
    local prev_c = inputs[L*2]
    local prev_h = inputs[L*2+1]
    -- the input to this layer
    if L == 1 then
      local word_vecs
      if model == 'enc' then
	word_vecs = nn.LookupTable(data.source_size, input_size)
      else
	word_vecs = nn.LookupTable(data.target_size, input_size)
      end
      word_vecs.name = 'word_vecs' .. name
      x = word_vecs(inputs[1]) -- batch_size x word_vec_size
      input_size_L = input_size
    else
      x = outputs[(L-1)*2]
      input_size_L = rnn_size
      if dropout > 0 then
        x = nn.Dropout(dropout, nil, false):usePrealloc(
	  nameL.."dropout", {{opt.max_batch_l, input_size_L}})(x)
      end
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):usePrealloc(nameL.."i2h-reuse",
                                                                  {{opt.max_batch_l, input_size_L}},
                                                                  {{opt.max_batch_l, 4 * rnn_size}})(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size, false):usePrealloc(nameL.."h2h-reuse",
								     {{opt.max_batch_l, rnn_size}},
								     {{opt.max_batch_l, 4 * rnn_size}})(prev_h)
    local all_input_sums = nn.CAddTable():usePrealloc(nameL.."allinput",
                                                      {{opt.max_batch_l, 4*rnn_size},
							{opt.max_batch_l, 4*rnn_size}},
                                                      {{opt.max_batch_l, 4 * rnn_size}})({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2):usePrealloc(nameL.."reshapesplit",
                                                        {{opt.max_batch_l, 4, rnn_size}})(reshaped):split(4)
    local in_gate = nn.Sigmoid():usePrealloc(nameL.."G1-reuse",{RnnD})(n1)
    local forget_gate = nn.Sigmoid():usePrealloc(nameL.."G2-reuse",{RnnD})(n2)
    local out_gate = nn.Sigmoid():usePrealloc(nameL.."G3-reuse",{RnnD})(n3)
    local in_transform = nn.Tanh():usePrealloc(nameL.."G4-reuse",{RnnD})(n4)
    local next_c = nn.CAddTable():usePrealloc(nameL.."G5a",{RnnD,RnnD})({
        nn.CMulTable():usePrealloc(nameL.."G5b",{RnnD,RnnD})({forget_gate, prev_c}),
        nn.CMulTable():usePrealloc(nameL.."G5c",{RnnD,RnnD})({in_gate, in_transform})})
    local next_h = nn.CMulTable():usePrealloc(nameL.."G5d",{RnnD,RnnD})
    ({out_gate, nn.Tanh():usePrealloc(nameL.."G6-reuse",{RnnD})(next_c)})
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  return nn.gModule(inputs, outputs)
end

function make_generator(data, opt)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  local attn_layer = make_decoder_attn(data, opt)
  attn_layer.name = 'attn_layer'
  local attn_out = attn_layer({inputs[2], inputs[1]})
  if opt.dropout > 0 then
    attn_out = nn.Dropout(opt.dropout, nil, false)(attn_out)
  end
  local output = nn.LogSoftMax()(nn.Linear(opt.rnn_size, data.target_size)(attn_out))
  return nn.gModule(inputs, {output})
end

function make_decoder_attn(data, opt)
  local inputs = {}
  local context_size = opt.rnn_size
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  local context = inputs[2]
  local combined_score = nn.MM()({context, nn.Replicate(1,3)(
				    nn.Linear(opt.rnn_size, opt.rnn_size, false)(inputs[1]))})
  local attn_dist, attn
  if opt.attn == 'crf' then
    local combined_score2 = nn.MulConstant(0)(combined_score)        
    local crf = nn.CRF(2)
    crf.name = 'crf'
    local crf_score = crf(nn.JoinTable(3)({combined_score, combined_score2}))
    attn_dist = nn.Sequential()
    attn_dist.name = 'attn_dist'
    attn_dist:add(nn.Exp()):add(nn.Narrow(2,2,-1)):add(nn.Sum(4))
    attn_dist:add(nn.Select(3,1))
    attn = attn_dist(crf_score)
    attn = nn.MulConstant(opt.lambda)(nn.Normalize(1)(attn))
  else
    combined_score = nn.Sum(3)(combined_score)
    if opt.attn == 'softmax' then
      attn_dist = nn.SoftMax()
    elseif opt.attn == 'sigmoid' then
      attn_dist = nn.Sigmoid()
    end  
    attn_dist.name = 'attn_dist'
    attn = attn_dist(combined_score)
  end  
  attn = nn.Replicate(1,2)(attn) -- batch_l x 1 x source_l
  -- apply attention to context
  local context_combined = nn.MM():usePrealloc(
    "dec_attn_mm2",{{opt.max_batch_l, 1, opt.max_sent_l_src},
      {opt.max_batch_l, opt.max_sent_l_src, context_size}},
    {{opt.max_batch_l, 1, context_size}})({attn, context}) -- batch_l x 1 x rnn_size
  context_combined = nn.Sum(2):usePrealloc(
    "dec_attn_sum", {{opt.max_batch_l, 1, context_size}},
    {{opt.max_batch_l, context_size}})(context_combined) -- batch_l x rnn_size
  context_combined = nn.JoinTable(2):usePrealloc(
    "dec_attn_jointable", {{opt.max_batch_l, context_size},
      {opt.max_batch_l, opt.rnn_size}})({context_combined, inputs[1]}) -- batch_l x rnn_size*2
  local context_output = nn.Tanh():usePrealloc("dec_attn_tanh",{{opt.max_batch_l,opt.rnn_size}})
  (nn.Linear(opt.rnn_size + context_size ,opt.rnn_size,false):usePrealloc(
     "dec_attn_linear", {{opt.max_batch_l,2*opt.rnn_size}})(context_combined))
  return nn.gModule(inputs, {context_output})
end

