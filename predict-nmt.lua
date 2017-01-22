require 'nn'
require 'string'
require 'hdf5'
require 'nngraph'
require 'models/models-nmt.lua'
stringx = require('pl.stringx')

cmd = torch.CmdLine()

-- file location
cmd:option('-model', '', [[Path to model .t7 file]])
cmd:option('-src_file', '',
	   [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the
                                       decoded sequence]])
cmd:option('-src_dict', 'data/nmt.src.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'data/nmt.targ.dict', [[Path to target vocabulary (*.targ.dict file)]])

-- beam search options
cmd:option('-beam', 5,[[Beam size]])
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer
                               than this then it will error out]])
cmd:option('-gpuid',  -1,[[ID of the GPU to use (-1 = use CPU)]])
opt = cmd:parse(arg)

function copy(orig)
   local orig_type = type(orig)
   local copy
   if orig_type == 'table' then
      copy = {}
      for orig_key, orig_value in pairs(orig) do
         copy[orig_key] = orig_value
      end
   else
      copy = orig
   end
   return copy
end

local StateAll = torch.class("StateAll")

function StateAll.initial(start)
   return {start}
end

function StateAll.advance(state, token)
   local new_state = copy(state)
   table.insert(new_state, token)
   return new_state
end

function StateAll.disallow(out)
   local bad = {1, 3} -- 1 is PAD, 3 is BOS
   for j = 1, #bad do
      out[bad[j]] = -1e9
   end
end

function StateAll.same(state1, state2)
   for i = 2, #state1 do
      if state1[i] ~= state2[i] then
         return false
      end
   end
   return true
end

function StateAll.next(state)
   return state[#state]
end

function StateAll.heuristic(state)
   return 0
end

function StateAll.print(state)
   for i = 1, #state do
      io.write(state[i] .. " ")
   end
   print()
end


-- Convert a flat index to a row-column tuple.
function flat_to_rc(v, flat_index)
   local row = math.floor((flat_index - 1) / v:size(2)) + 1
   return row, (flat_index - 1) % v:size(2) + 1
end

function generate_beam(model, initial, K, max_sent_l, source, gold)
   --reset decoder initial states
   local n
   local source_l = math.min(source:size(1), opt.max_sent_l)
   local attn_argmax = {}   -- store attn weights
   if opt.length_restriction == 0 then
      n = max_sent_l
   else
      n = source_l + 5
   end   
   
  -- Backpointer table.
   local prev_ks = torch.LongTensor(n, K):fill(1)
   -- Current States.
   local next_ys = torch.LongTensor(n, K):fill(1)
   -- Current Scores.
   local scores = torch.FloatTensor(n, K)
   scores:zero()
   
   local states = {} -- store predicted word idx
   states[1] = {}
   for k = 1, 1 do
      table.insert(states[1], initial)
      next_ys[1][k] = State.next(initial)
   end

   local source_input = source:view(source_l, 1):cuda()
   local rnn_state_enc = {}
   for i = 1, #init_fwd_enc do
      table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
   end   
   local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size
   
   for t = 1, source_l do
     local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
      local out = encoder:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:copy(out[#out])
   end
   context = context:expand(K, source_l, model_opt.context_size):contiguous()
   rnn_state_dec = {}
   for i = 1, #init_fwd_dec do
      table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
   end
   out_float = torch.FloatTensor()   
   local i = 1
   local done = false
   local max_score = -1e9
   local found_eos = false
   while (not done) and (i < n) do
      i = i+1
      states[i] = {}
      
      local decoder_input1 = next_ys:narrow(1,i-1,1):squeeze()
      if opt.beam == 1 then
	decoder_input1 = torch.LongTensor({decoder_input1})
      end	
      local decoder_input = {decoder_input1, table.unpack(rnn_state_dec)}
      local out_decoder = decoder:forward(decoder_input)
      local out = generator:forward({context, out_decoder[#out_decoder]}) -- K x vocab_size      
      rnn_state_dec = {} -- to be modified later
      for j = 1, #out_decoder do
	 table.insert(rnn_state_dec, out_decoder[j])
      end
      
      out_float:resize(out:size()):copy(out)
      for k = 1, K do
	 State.disallow(out_float:select(1, k))
	 out_float[k]:add(scores[i-1][k])
      end
      -- All the scores available.
       local flat_out = out_float:view(-1)
       if i == 2 then
          flat_out = out_float[1] -- all outputs same for first batch
       end       
       for k = 1, K do
             local score, index = flat_out:max(1)
             local score = score[1]
             local prev_k, y_i = flat_to_rc(out_float, index[1])
             states[i][k] = State.advance(states[i-1][prev_k], y_i)	     
	     prev_ks[i][k] = prev_k
	     next_ys[i][k] = y_i
	     scores[i][k] = score
	     flat_out[index[1]] = -1e9
       end
       for j = 1, #rnn_state_dec do
	  rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, prev_ks[i]))
       end
       end_hyp = states[i][1]
       end_score = scores[i][1]
       if end_hyp[#end_hyp] == END then
	  done = true
	  found_eos = true
       else
	  for k = 1, K do
	     local possible_hyp = states[i][k]
	     if possible_hyp[#possible_hyp] == END then
		found_eos = true
		if scores[i][k] > max_score then
		   max_hyp = possible_hyp
		   max_score = scores[i][k]
		end
	     end	     
	  end	  
       end       
   end
   local gold_score = 0
   if end_score > max_score or not found_eos then
      max_hyp = end_hyp
      max_score = end_score
   end
   return max_hyp
end

function idx2key(file)   
   local f = io.open(file,'r')
   local t = {}
   for line in f:lines() do
      local c = {}
      for w in line:gmatch'([^%s]+)' do
	 table.insert(c, w)
      end
      t[tonumber(c[2])] = c[1]
   end   
   return t
end

function flip_table(u)
   local t = {}
   for key, value in pairs(u) do
      t[value] = key
   end
   return t   
end

function sent2wordidx(sent, word2idx, start_symbol)
   local t = {}
   local u = {}
   for word in sent:gmatch'([^%s]+)' do
      local idx = word2idx[word] or UNK 
      table.insert(t, idx)
      table.insert(u, word)
   end
   return torch.LongTensor(t), u
end

function wordidx2sent(sent, idx2word, skip_end)
   local t = {}
   local start_i, end_i
   skip_end = skip_start_end or true
   if skip_end then
      end_i = #sent-1
   else
      end_i = #sent
   end   
   for i = 2, end_i do -- skip START and END
      if sent[i] == UNK then
	table.insert(t, idx2word[sent[i]])
      else
	 table.insert(t, idx2word[sent[i]])	 
      end           
   end
   return table.concat(t, ' ')
end


function strip(s)
   return s:gsub("^%s+",""):gsub("%s+$","")
end

function main()
   -- some globals
   PAD = 1; UNK = 2; START = 3; END = 4
   MAX_SENT_L = opt.max_sent_l
   assert(path.exists(opt.src_file), 'src_file does not exist')
   assert(path.exists(opt.model), 'model does not exist')
   
   -- parse input params
   opt = cmd:parse(arg)
   if opt.gpuid >= 0 then
      require 'cutorch'
      require 'cunn'
   end      
   print('loading ' .. opt.model .. '...')
   checkpoint = torch.load(opt.model)
   print('done!')

   if opt.replace_unk == 1 then
      phrase_table = {}
      if path.exists(opt.srctarg_dict) then
	 local f = io.open(opt.srctarg_dict,'r')
	 for line in f:lines() do
	    local c = line:split("|||")
	    phrase_table[strip(c[1])] = c[2]
	 end
      end      
   end

   -- load model and word2idx/idx2word dictionaries
   model, model_opt = checkpoint[1], checkpoint[2]
   encoder, decoder, generator = table.unpack(model)
   --   print(model_opt)
   for u, v in pairs(model_opt) do
     if opt[u] == nil then
       opt[u] = v
     end     
   end
   if model_opt.cudnn == 1 then
       require 'cudnn'
    end
   
   idx2word_src = idx2key(opt.src_dict)
   word2idx_src = flip_table(idx2word_src)
   idx2word_targ = idx2key(opt.targ_dict)
   word2idx_targ = flip_table(idx2word_targ)
   -- load gold labels if it exists
   if path.exists(opt.targ_file) then
      print('loading GOLD labels at ' .. opt.targ_file)
      gold = {}
      local file = io.open(opt.targ_file, 'r')
      for line in file:lines() do
	 table.insert(gold, line)
      end
   end
   if opt.gpuid >= 0 then
      cutorch.setDevice(opt.gpuid)
      for i = 1, #model do
	model[i]:double():cuda()
	model[i]:evaluate()
      end
   end
   context_proto = torch.zeros(1, MAX_SENT_L, model_opt.context_size)
   local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
   local h_init_enc = torch.zeros(1, model_opt.rnn_size) 
   if opt.gpuid >= 0 then
      h_init_enc = h_init_enc:cuda()      
      h_init_dec = h_init_dec:cuda()
      cutorch.setDevice(opt.gpuid)
      context_proto = context_proto:cuda()
   end
   init_fwd_enc = {}
   init_fwd_dec = {} -- initial context   
   for L = 1, model_opt.num_layers do
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
      table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state      
   end
   State = StateAll
   
   local sent_id = 0
   pred_sents = {}
   local file = io.open(opt.src_file, "r")
   local out_file = io.open(opt.output_file,'w')
   timer = torch.Timer()
   local start_time = timer:time().real
   for line in file:lines() do
      sent_id = sent_id + 1
      print('SENT ' .. sent_id .. ': ' ..line)
      local source, source_str = sent2wordidx(line, word2idx_src, model_opt.start_symbol)
      state = State.initial(START)
      pred = generate_beam(model, state, opt.beam, MAX_SENT_L, source)
      pred_sent = wordidx2sent(pred, idx2word_targ,true)
      out_file:write(pred_sent .. '\n')      
      print('PRED ' .. sent_id .. ': ' .. pred_sent)
      if gold ~= nil then
	 print('GOLD ' .. sent_id .. ': ' .. gold[sent_id])
      end
      print('')
      time_taken = timer:time().real - start_time
   end
   out_file:close()
end
main()

