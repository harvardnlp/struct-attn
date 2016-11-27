require 'nn'
require 'EisnerCRF.lua'

function is_spanning(parse)
   -- returns true if tree is spanning
   local n = parse:size(1)
   local d = {}
   for i = 1, n do
      local h = parse[i]
      if i == h then
	 return false
      end
      if d[h] == nil then
	 d[h] = {}
      end
      table.insert(d[h], i)
   end
   local stack = {1}
   local seen = {}
   while #stack > 0 do
      local cur = table.remove(stack, 1)
      if seen[cur] == 1 then
	 return false
      end
      seen[cur] = 1
      if d[cur] ~= nil then
	 for i = 1, #d[cur] do
	    table.insert(stack, 1, d[cur][i])
	 end
      end      
   end
   local l = 0
   for i, j in pairs(seen) do
      l = l + 1
   end
   if l ~= n then
      return false
   end
   return true
end

function is_projective(parse)
   -- returns true if tree is projective
   local n = parse:size(1)
   for i = 1, n do
      local h = parse[i]
      for j = 1, n do
	 local h2 = parse[j]
	 if i ~= j then
	    if i < h then
	       if (i < j and j < h and h < h2) or (i < h2 and h2 < h and h < j) or (
		  j < i and i < h2 and h2 < h) or (h2 < i and i < j and j < h) then
		  return false
	       end
	    end
	    if h < i then
	       if (h < j and j < i and i < h2) or (h < h2 and h2 < i and i < j) or (
		  j < h and h < h2 and h2 < i) or (h2 < h and h < j and j < i) then
		  return false
	       end	       
	    end	    
	 end	 
      end
   end
   return true   	 
end

function all_parses(n)
   -- all valid parse trees with n nodes
   local code = 'parses = {}; parse = torch.zeros(' .. n .. ')'
   for i = 1, n-1 do
      local v = 'i' .. i
      code = code .. ' for ' .. v .. ' = 1, ' .. n .. ' do \n'
      code = code .. ' parse[' .. i+1 .. '] = ' .. v .. '\n'      
   end
   code = code .. ' if parse:eq(1):sum() > 0 and is_spanning(parse) and is_projective(parse) then \n'
   code = code .. ' table.insert(parses, parse:clone()) \n'
   for i = 1, n do
      code = code .. ' end \n'
   end
   return code
end

function score_tree(scores, parse)
   -- scores a parse tree
   local n = parse:size(1)
   local s = 0 
   for i = 2, n do
      s = s + scores[parse[i]][i]
   end
   return s   
end

function calc_manual(scores, parses)
   -- manually get best tree and partition function and marginals
   local m = -math.huge
   local z = 0
   local n = parses[1]:size()[1]
   local log_marginal = torch.Tensor(n,n):fill(-math.huge)
   
   for i = 1, #parses do
      local parse = parses[i]
      local s = score_tree(scores, parse)
      if s > m then
	 m = s
	 best = parse
      end
      z = z + math.exp(s)
      for j = 1, n-1 do
	 for k = j+1, n do
	    if parse[k] == j then -- j ==> k	       
	       log_marginal[j][k] = math.log(math.exp(log_marginal[j][k]) + math.exp(s))
	    end
	    if parse[j] == k then -- k ==> j
	       log_marginal[k][j] = math.log(math.exp(log_marginal[k][j]) + math.exp(s))
	    end	    
	 end
      end      	 
   end
   return m, best, z, log_marginal
end


function test_unit(n, b)
   local m = nn.EisnerCRF()
   local timer = torch.Timer()
   local dp_time = 0
   for i = 3, n do
      local code = all_parses(i)
      loadstring(code)()
      print(string.format('Testing n: %d, possible graphs: %d',i, #parses))
      local scores = torch.randn(b,i,i)
      local start_time = timer:time().real      
      local map_scores, map_trees = m:viterbi(scores)
      local marginals = m:forward(scores)
      local log_zs = m.log_inside[{{},1,i,RIGHT,COMPLETE}]      
      dp_time = dp_time + timer:time().real - start_time
      for j = 1, b do	 
	 local map_score, map_tree, z, log_marginal = calc_manual(scores[j], parses)
	 assert(torch.abs(map_scores[j] - map_score) < 1e-5, "scores doesn't match")
	 assert((map_trees[j] - map_tree):sum() == 0, "MAP tree doesn't match")
	 assert(torch.abs(math.exp(log_zs[j]) - z) < 1e-5, "inside --> partition doesn't match")
	 for k = 2, i do
	    assert(torch.abs(math.exp(m.log_outside[j][k][k][LEFT][COMPLETE]) - z) < 1e-5,
		   "outside --> partition doesn't match")
	    assert(torch.abs(math.exp(m.log_outside[j][k][k][RIGHT][COMPLETE]) - z) < 1e-5,
		   "outside --> partition doesn't match")	    
	 end	 	 
	 assert((marginals[j] - log_marginal:add(-math.log(z)):exp()):abs():sum() < 1e-5,
	    "marginals don't match")
      end
      print("passed!")
   end
   print(string.format('Viterbi/Inside-Outside took %.2f seconds', dp_time))
end

function grad_check(n, b)
   local m = nn.EisnerCRF()
   local crit = nn.MSECriterion()
   torch.manualSeed(3435)
   local x = torch.randn(b, n, n)
   local y = torch.randn(b, n, n)   
   local eps = 1e-8
   local pred = m:forward(x)
   local loss = crit:forward(pred,y)
   local dl_dpred = crit:backward(pred,y)
   local dl_dx = m:backward(x, dl_dpred)
   local C = m.log_inside
   local max_diff = -math.huge
   local min_diff = math.huge
   for i = 1, b do
      for s = 1, n-1 do
   	 for t = s+1, n do
   	    x[i][s][t] = x[i][s][t] + eps
   	    local loss1 = crit:forward(m:forward(x),y)
   	    x[i][s][t] = x[i][s][t] - 2*eps
   	    local loss2 = crit:forward(m:forward(x),y)
   	    local grad_est = (loss1 - loss2) / (2*eps)
   	    x[i][s][t] = x[i][s][t] + eps	    
	    if grad_est/dl_dx[i][s][t] > max_diff then
	       max_diff = grad_est/dl_dx[i][s][t]
	    end
	    if grad_est/dl_dx[i][s][t] < min_diff then
	       min_diff = grad_est/dl_dx[i][s][t]
	    end	    	    
   	    if s > 1 then
   	       x[i][t][s] = x[i][t][s] + eps
   	       local loss1 = crit:forward(m:forward(x),y)
   	       x[i][t][s] = x[i][t][s] - 2*eps
   	       local loss2 = crit:forward(m:forward(x),y)
   	       local grad_est = (loss1 - loss2) / (2*eps)
   	       x[i][t][s] = x[i][t][s] + eps
	       if grad_est/dl_dx[i][t][s] > max_diff then
		  max_diff = grad_est/dl_dx[i][t][s]
	       end
	       if grad_est/dl_dx[i][t][s] < min_diff then
		  min_diff = grad_est/dl_dx[i][t][s]
	       end	    	       
   	    end
   	 end	 
      end
   end
   print(string.format("for length %d and batch size %d, min ratio: %.6f, max_ratio: %.6f",
		       n, b, min_diff, max_diff))
end

print("testing viterbi and marginals...")
test_unit(6, 100)
print("testing gradients...")
grad_check(6, 10)
