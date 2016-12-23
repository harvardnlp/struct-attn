-- globals
LEFT = 1
RIGHT = 2
INCOMPLETE = 1
COMPLETE = 2

function isnan(i)
   return (i == i and i or -math.huge) 
end

function fixsign(i)
   return (i == 0 and 1 or i)
end

local EisnerCRF, parent = torch.class('nn.EisnerCRF', 'nn.Module')

function EisnerCRF:__init()
   self.C = torch.Tensor() -- chart scores for viterbi (batch x n x n x 2 x 2)
   self.heads = torch.Tensor() -- MAP output (batch x n)
   self.bp = torch.Tensor() -- backpointer for viterbi (batch x n x n x 2 x 2)
   self.log_inside = torch.Tensor() -- inside potentials (batch x n x n x 2 x 2)
   self.log_outside = torch.Tensor() -- outside potentials (batch x n x n x 2 x 2)
   
   self.output = torch.Tensor() -- marginals (batch x n x n)
   self.log_marginals = torch.Tensor() -- log marginals (batch x n x n)
   self.gradInput = torch.Tensor() -- grad input (batch x n x n)

   self.tmp = torch.Tensor() -- temp tensors for intermediate calcs
   self.tmp2 = torch.Tensor()
   self.tmp3 = torch.Tensor() 
   self.ge = torch.Tensor()
   
   self.log_grad_marginal = torch.Tensor() -- p * dL/dp
   self.log_grad_marginal_sign = torch.Tensor() -- p * dL/dp   
   self.log_grad_input = torch.Tensor() -- log grad input
   self.log_grad_input_sign = torch.Tensor()
   self.log_grad_output = torch.Tensor() -- log grad output
   self.log_grad_output_sign = torch.Tensor()
   self.log_grad_inside = torch.Tensor() -- log grad wrt inside terms
   self.log_grad_inside_sign = torch.Tensor()    
   self.log_grad_outside = torch.Tensor() -- log grad wrt outside terms
   self.log_grad_outside_sign = torch.Tensor()

   self.cuda_mod = 0
   
end

function EisnerCRF:logsumexp(x, y, d)
   -- does logsumexp over dimension d of tensor y and puts the result into x
   self.tmp3:max(y,d)
   x:copy(y:add(-1, self.tmp3:expandAs(y)):exp():sum(d):log():add(self.tmp3))
end

function EisnerCRF:logadd(x, y, x_sign, y_sign)
   -- returns log (exp(x) + exp(y)) and puts the result in x (and x_sign)
   if x_sign == nil then
      if self.cuda_mod == 0 then
	 self.tmp3:cmax(x, y)   	 
	 x:add(-self.tmp3):exp():add(y:add(-self.tmp3):exp()):log():add(self.tmp3)
      else
	 x.THCUDAMOD.LogSpace_add_inplace(x:cdata(), y:cdata())
      end      
   else
      self.ge:ge(x,y)
      if self.cuda_mod == 0 then
	 self.tmp3:cmax(x, y)   	       
	 x:cmin(x,y):add(-1, self.tmp3):exp():cmul(x_sign):cmul(y_sign):log1p():add(self.tmp3) 
	 x_sign:cmul(self.ge)
	 self.ge:add(-1):mul(-1)
	 self.ge:cmul(y_sign)
	 x_sign:add(self.ge)
	 x:apply(isnan)
      else
	 x.THCUDAMOD.SignedLogSpace_add_inplace(x:cdata(), y:cdata(),
						x_sign:cdata(), y_sign:cdata(),
						self.tmp3:cdata(), self.ge:cdata())
      end
   end      
end

function EisnerCRF:updateOutput(input)
   local batch = input:size(1)
   local n = input:size(2)
   self.input_expand = input:view(batch, n, n, 1):expand(batch, n, n, n)
   self.tmp:resize(batch, n)
   self.tmp2:resize(batch, n)
   self.tmp3:resize(batch)
   self.ge:resize(batch)
   
   self.output:resize(batch, n, n)
   self.log_marginals:resize(batch, n, n):fill(-math.huge)
   self:inside(input)
   self:outside(input)
   -- calculate log marginals 
   for s = 1, n-1 do
      for t = s+1, n do
   	 self.log_marginals[{{},s,t}]:copy(self.log_outside[{{}, s, t, RIGHT, INCOMPLETE}]):add(
   	    self.log_inside[{{}, s, t, RIGHT, INCOMPLETE}])
   	 if s > 1 then -- root has no head
   	    self.log_marginals[{{},t,s}]:copy(self.log_outside[{{}, s, t, LEFT, INCOMPLETE}]):add(
   	       self.log_inside[{{}, s, t, LEFT, INCOMPLETE}])
   	 end	    
      end
   end
   
   self.log_z = self.log_inside[{{}, 1, n, RIGHT, COMPLETE}]:contiguous():view(batch, 1, 1)
   self.log_marginals:add(-1, self.log_z:expand(batch, n, n))
   self.output:copy(self.log_marginals):exp()
   return self.output
end

function EisnerCRF:inside(input)
   -- calculate inside potentials
   -- input is a (batch x n x n) of scores where
   -- input[b][s][t] is the score for connecting s ==> t
   -- inside potentials are in log_inside[b][s][t][d][c] with the following notation:
   -- b \in [1, ..., batch] = batch index 
   -- s \in [1, ..., n] = left most index in span
   -- t \in [1, ..., n] = right most index in span
   -- d \in  {1, 2} = {<==, ==>}
   -- c \in  {1, 2} = {incomplete, complete}
   
   local batch = input:size(1)
   local n = input:size(2)
   self.log_inside:resize(batch, n, n, 2, 2):fill(-math.huge)
   for i = 1, n do
      self.log_inside[{{}, i, i, {}, COMPLETE}]:zero()
   end   
   for k = 1, n do
      for s = 1, n do	 
	 local t = s+k
	 if t > n then break end
	 local l = t-s
	 -- left incomplete tree potentials	 
	 if s > 1 then
	    self.tmp[{{}, {s, t-1}}]:copy(
	       self.log_inside[{{}, s, {}, RIGHT, COMPLETE}]:narrow(2, s, l)):add(
	       self.log_inside[{{}, {}, t, LEFT, COMPLETE}]:narrow(2, s+1, l)):add(
	       self.input_expand[{{}, t, s, {1, l}}])
	    for u = s, t-1 do
	       self:logadd(self.log_inside[{{}, s, t, LEFT, INCOMPLETE}], self.tmp[{{}, u}])
	    end	    
	 end	 
	 -- right incomplete tree potentials
	 self.tmp[{{}, {s, t-1}}]:copy(
	    self.log_inside[{{}, s, {}, RIGHT, COMPLETE}]:narrow(2, s, l)):add(
	    self.log_inside[{{}, {}, t, LEFT, COMPLETE}]:narrow(2, s+1, l)):add(
	    self.input_expand[{{}, s, t, {1, l}}])
	 for u = s, t-1 do
	    self:logadd(self.log_inside[{{}, s, t, RIGHT, INCOMPLETE}], self.tmp[{{}, u}])
	 end	 
	 -- left complete tree potentials
	 if s > 1 then
	    self.tmp[{{}, {s, t-1}}]:copy(
	       self.log_inside[{{}, s, {}, LEFT, COMPLETE}]:narrow(2, s, l)):add(
	       self.log_inside[{{}, {}, t, LEFT, INCOMPLETE}]:narrow(2, s, l))		  
	    for u = s, t-1 do
	       self:logadd(self.log_inside[{{}, s, t, LEFT, COMPLETE}], self.tmp[{{}, u}])
	    end
	 end	 
	 -- right complete tree potentials
	 self.tmp[{{}, {s+1, t}}]:copy(
	    self.log_inside[{{}, s, {}, RIGHT, INCOMPLETE}]:narrow(2, s+1, l)):add(
	    self.log_inside[{{}, {}, t, RIGHT, COMPLETE}]:narrow(2, s+1, l))
	 for u = s+1, t do
	    self:logadd(self.log_inside[{{}, s, t, RIGHT, COMPLETE}], self.tmp[{{}, u}])
	 end	 
      end
   end
end

function EisnerCRF:outside(input)
   -- calculate outside potentials
   -- input is a (batch x n x n) of scores where
   -- input[b][s][t] is the score for connecting s ==> t
   -- outside potentials are in log_outside[b][s][t][d][c] with the following notation:
   -- b \in [1, ..., batch] = batch index 
   -- s \in [1, ..., n] = left most index in span
   -- t \in [1, ..., n] = right most index in span
   -- d \in  {1, 2} = {<==, ==>}
   -- c \in  {1, 2} = {incomplete, complete}
   
   local batch = input:size(1)
   local n = input:size(2)
   self.log_outside:resize(batch, n, n, 2, 2):fill(-math.huge)
   self.log_outside[{{}, 1, n, RIGHT, COMPLETE}]:zero()   
   local log_outside_expand = self.log_outside:view(
      batch, n, n, 2, 2, 1):expand(batch, n, n, 2, 2, n)
   for k = n, 1, -1 do
      for s = 1, n do	 
	 local t = s+k
	 if t > n then break end
	 local l = t-s	 
	 -- left complete
	 if s > 1 then	    
	    self.tmp[{{}, {s, t-1}}]:copy(
	       log_outside_expand[{{}, s, t, LEFT, COMPLETE, {s, t-1}}]):add(
	       self.log_inside[{{}, {}, t, LEFT, INCOMPLETE}]:narrow(2, s, l))
	    for u = s, t-1 do
	       self:logadd(self.log_outside[{{}, s, u, LEFT, COMPLETE}], self.tmp[{{}, u}])
	    end
	    self.tmp[{{}, {s, t-1}}]:copy(
	       log_outside_expand[{{}, s, t, LEFT, COMPLETE, {s, t-1}}]):add(
	       self.log_inside[{{}, s, {}, LEFT, COMPLETE}]:narrow(2, s, l))
	    for u = s, t-1 do
	       self:logadd(self.log_outside[{{}, u, t, LEFT, INCOMPLETE}], self.tmp[{{}, u}])
	    end	    
	 end
      	 -- -- right complete	 
   	 self.tmp[{{}, {s+1, t}}]:copy(
	    log_outside_expand[{{}, s, t, RIGHT, COMPLETE, {s+1, t}}]):add(
   	    self.log_inside[{{}, {}, t, RIGHT, COMPLETE}]:narrow(2, s+1, l))
   	 for u = s+1, t do
   	    self:logadd(self.log_outside[{{}, s, u, RIGHT, INCOMPLETE}], self.tmp[{{}, u}])
   	 end
   	 self.tmp[{{}, {s+1, t}}]:copy(
	    log_outside_expand[{{}, s, t, RIGHT, COMPLETE, {s+1, t}}]):add(
   	    self.log_inside[{{}, s, {}, RIGHT, INCOMPLETE}]:narrow(2, s+1, l))
   	 for u = s+1, t do
   	    self:logadd(self.log_outside[{{}, u, t, RIGHT, COMPLETE}], self.tmp[{{}, u}])
   	 end	 
      	 -- left incomplete
   	 if s > 1 then	    
   	    self.tmp2[{{}, {s, t-1}}]:copy(
	       log_outside_expand[{{}, s, t, LEFT, INCOMPLETE, {s, t-1}}]):add(
   	       self.input_expand[{{}, t, s, {1, l}}])
   	    self.tmp[{{}, {s, t-1}}]:copy(self.tmp2[{{}, {s, t-1}}]):add(
   		  self.log_inside[{{}, {}, t, LEFT, COMPLETE}]:narrow(2, s+1, l))
   	    for u = s, t-1 do
   	       self:logadd(self.log_outside[{{}, s, u, RIGHT, COMPLETE}], self.tmp[{{}, u}])
   	    end
   	    self.tmp[{{}, {s, t-1}}]:copy(self.tmp2[{{}, {s, t-1}}]):add(
   	       self.log_inside[{{}, s, {}, RIGHT, COMPLETE}]:narrow(2, s, l))
   	    for u = s, t-1 do
   	       self:logadd(self.log_outside[{{}, u+1, t, LEFT, COMPLETE}], self.tmp[{{}, u}])
   	    end
   	 end	 
      	 -- right incomplete
   	 self.tmp2[{{}, {s, t-1}}]:copy(
	    log_outside_expand[{{}, s, t, RIGHT, INCOMPLETE, {s, t-1}}]):add(
   	    self.input_expand[{{}, s, t, {1, l}}])
   	 self.tmp[{{}, {s, t-1}}]:copy(self.tmp2[{{}, {s, t-1}}]):add(
   	    self.log_inside[{{}, {}, t, LEFT, COMPLETE}]:narrow(2, s+1, l))
   	 for u = s, t-1 do
   	    self:logadd(self.log_outside[{{}, s, u, RIGHT, COMPLETE}], self.tmp[{{}, u}])
   	 end
   	 self.tmp[{{}, {s, t-1}}]:copy(self.tmp2[{{}, {s, t-1}}]):add(
   	    self.log_inside[{{}, s, {}, RIGHT, COMPLETE}]:narrow(2, s, l))
   	 for u = s, t-1 do
   	    self:logadd(self.log_outside[{{}, u+1, t, LEFT, COMPLETE}], self.tmp[{{}, u}])
   	 end
      end
   end   
end

function EisnerCRF:updateGradInput(input, gradOutput)
   local batch = input:size(1)
   local n = input:size(2)   
   self.gradInput:resizeAs(input)
   self.log_grad_input:resizeAs(input):fill(-math.huge)
   self.log_grad_input_sign:resizeAs(input):fill(1)
   
   -- get log grad output
   self.log_grad_output:resizeAs(gradOutput):copy(gradOutput):abs():log()
   self.log_grad_output_sign:resizeAs(gradOutput):sign(gradOutput)

   -- log_grad_marginal = log(p*gradOutput)
   self.log_grad_marginal:resizeAs(self.output):copy(self.log_marginals)
   self.log_grad_marginal:add(self.log_grad_output)
   self.log_grad_marginal_sign:resizeAs(self.output):copy(self.log_grad_output_sign)
   
   -- backprop through outside/inside
   -- initialize grads/signs for inside/outside charts
   self.log_grad_inside:resize(batch, n, n, 2, 2):fill(-math.huge)
   self.log_grad_inside_sign:resize(batch, n, n, 2, 2):fill(1)
   self.log_grad_outside:resize(batch, n, n, 2, 2):fill(-math.huge)
   self.log_grad_outside_sign:resize(batch, n, n, 2, 2):fill(1)

   -- local start = self.timer:time().real
   -- backprop log_grad_marginal to left/right incompletes that produced the marginals
   for s = 1, n-1 do
      for t = s+1, n do
   	 self.log_grad_inside[{{}, s, t, RIGHT, INCOMPLETE}]:copy(
   	    self.log_grad_marginal[{{}, s, t}])
   	 self.log_grad_inside_sign[{{}, s, t, RIGHT, INCOMPLETE}]:copy(
   	    self.log_grad_marginal_sign[{{}, s, t}])	 

   	 self.log_grad_outside[{{}, s, t, RIGHT, INCOMPLETE}]:copy(
   	    self.log_grad_marginal[{{}, s, t}])
   	 self.log_grad_outside_sign[{{}, s, t, RIGHT, INCOMPLETE}]:copy(
   	    self.log_grad_marginal_sign[{{}, s, t}])
	 
   	 self:logadd(self.log_grad_inside[{{}, 1, n, RIGHT, COMPLETE}],
   	 	     self.log_grad_marginal[{{}, s, t}],
   	 	     self.log_grad_inside_sign[{{}, 1, n, RIGHT, COMPLETE}],
   	 	     torch.mul(self.log_grad_marginal_sign[{{}, s, t}], -1)) -- we are subtracting
   	 if s > 1 then -- root has no head
   	    self.log_grad_inside[{{}, s, t, LEFT, INCOMPLETE}]:copy(
   	       self.log_grad_marginal[{{}, t, s}])	    
   	    self.log_grad_inside_sign[{{}, s, t, LEFT, INCOMPLETE}]:copy(
   	       self.log_grad_marginal_sign[{{}, t, s}])
	    
   	    self.log_grad_outside[{{}, s, t, LEFT, INCOMPLETE}]:copy(
   	       self.log_grad_marginal[{{}, t, s}])
   	    self.log_grad_outside_sign[{{}, s, t, LEFT, INCOMPLETE}]:copy(
   	       self.log_grad_marginal_sign[{{}, t, s}])
	    
	    self:logadd(self.log_grad_inside[{{}, 1, n, RIGHT, COMPLETE}],
	    		self.log_grad_marginal[{{}, t, s}],
	    		self.log_grad_inside_sign[{{}, 1, n, RIGHT, COMPLETE}],
	    		torch.mul(self.log_grad_marginal_sign[{{}, t, s}], -1)) -- we are subtracting
   	 end	    
      end
   end
   self:outside_backward(input)
   self:inside_backward(input)
   self.gradInput:copy(self.log_grad_input):exp():cmul(self.log_grad_input_sign)
   return self.gradInput
end

function EisnerCRF:inside_backward(input)
   -- backward step for the inside algorithm
   local batch = input:size(1)
   local n = input:size(2)
   local grad, grad_sign
   for k = n, 1, -1 do
      for s = 1, n do	 
	 local t = s+k
	 if t > n then break end
	 local l = t-s
      	 -- right_complete(s,t) gives grads to right_incomplete(s,u) and right_complete(u,t)
	 grad = self.log_grad_inside[{{}, s, t, RIGHT, COMPLETE}]:add(
	       -1, self.log_inside[{{}, s, t, RIGHT, COMPLETE}])
	 self.tmp2[{{}, 1}]:copy(grad)	 
	 grad_sign = self.log_grad_inside_sign[{{}, s, t, RIGHT, COMPLETE}]
	 self.tmp[{{}, {s+1, t}}]:copy(
	    self.log_inside[{{}, s, {}, RIGHT, INCOMPLETE}]:narrow(2, s+1, l)):add(
	    self.log_inside[{{}, {}, t, RIGHT, COMPLETE}]:narrow(2, s+1, l)):add(
	    self.tmp2:expand(batch, l))
      	 for u = s+1, t do	    
	    self:logadd(self.log_grad_inside[{{}, s, u, RIGHT, INCOMPLETE}], self.tmp[{{}, u}],
			self.log_grad_inside_sign[{{}, s, u, RIGHT, INCOMPLETE}], grad_sign)
	    self:logadd(self.log_grad_inside[{{}, u, t, RIGHT, COMPLETE}], self.tmp[{{}, u}],
			self.log_grad_inside_sign[{{}, u, t, RIGHT, COMPLETE}], grad_sign)
      	 end	 	 	 
	 -- left_complete(s,t) gives grads to left_complete(s,u) and left_incomplete(u,t)
	 if s > 1 then
	    grad = self.log_grad_inside[{{}, s, t, LEFT, COMPLETE}]:add(
		  -1, self.log_inside[{{}, s, t, LEFT, COMPLETE}])
	    self.tmp2[{{}, 1}]:copy(grad)	 	    
	    grad_sign = self.log_grad_inside_sign[{{}, s, t, LEFT, COMPLETE}]
	    self.tmp[{{}, {s, t-1}}]:copy(
	       self.log_inside[{{}, s, {}, LEFT, COMPLETE}]:narrow(2, s, l)):add(
	       self.log_inside[{{}, {}, t, LEFT, INCOMPLETE}]:narrow(2, s, l)):add(
	       self.tmp2:expand(batch, l))	    
	    for u = s, t-1 do
	       self:logadd(self.log_grad_inside[{{}, s, u, LEFT, COMPLETE}], self.tmp[{{}, u}],
			   self.log_grad_inside_sign[{{}, s, u, LEFT, COMPLETE}], grad_sign)
	       self:logadd(self.log_grad_inside[{{}, u, t, LEFT,INCOMPLETE}], self.tmp[{{}, u}],
			   self.log_grad_inside_sign[{{}, u, t, LEFT, INCOMPLETE}], grad_sign)
	    end
	    
	 end
      	 -- right_incomplete(s,t) gives grads to right_complete(s,u),
	 -- left_complete(u+1,t), p(s,t)
	 grad = self.log_grad_inside[{{}, s, t, RIGHT, INCOMPLETE}]:add(
	       -1, self.log_inside[{{}, s, t, RIGHT, INCOMPLETE}])
	 self.tmp2[{{}, 1}]:copy(grad)	 
	 grad_sign = self.log_grad_inside_sign[{{}, s, t, RIGHT, INCOMPLETE}]
	 self.tmp[{{}, {s, t-1}}]:copy(
	    self.log_inside[{{}, s, {}, RIGHT, COMPLETE}]:narrow(2, s, l)):add(
	    self.log_inside[{{}, {}, t, LEFT, COMPLETE}]:narrow(2, s+1, l)):add(
	    self.input_expand[{{}, s, t, {1, l}}]):add(self.tmp2:expand(batch, l))
      	 for u = s, t-1 do
	    self:logadd(self.log_grad_inside[{{}, s, u, RIGHT, COMPLETE}], self.tmp[{{}, u}],
	 		self.log_grad_inside_sign[{{}, s, u, RIGHT, COMPLETE}], grad_sign)
	    self:logadd(self.log_grad_inside[{{}, u+1, t, LEFT, COMPLETE}], self.tmp[{{}, u}],
	 		self.log_grad_inside_sign[{{}, u+1, t, LEFT, COMPLETE}], grad_sign)
	    self:logadd(self.log_grad_input[{{}, s, t}], self.tmp[{{}, u}],
			self.log_grad_input_sign[{{}, s, t}], grad_sign)	 	 
      	 end	 
      	 -- left_incomplete(s,t) gives grads to right_complete(s,u),
	 -- left_complete(u+1,t), p(t,s) 
	 if s > 1 then
	    grad = self.log_grad_inside[{{}, s, t, LEFT, INCOMPLETE}]:add(
		  -1, self.log_inside[{{}, s, t, LEFT, INCOMPLETE}])
	    self.tmp2[{{}, 1}]:copy(grad)	 	    
	    grad_sign = self.log_grad_inside_sign[{{}, s, t, LEFT, INCOMPLETE}]
	    self.tmp[{{}, {s, t-1}}]:copy(
	       self.log_inside[{{}, s, {}, RIGHT, COMPLETE}]:narrow(2, s, l)):add(
	       self.log_inside[{{}, {}, t, LEFT, COMPLETE}]:narrow(2, s+1, l)):add(
	       self.input_expand[{{}, t, s, {1, l}}]):add(self.tmp2:expand(batch, l))    
	    for u = s, t-1 do
	       self:logadd(self.log_grad_inside[{{}, s, u, RIGHT, COMPLETE}], self.tmp[{{}, u}],
	 		   self.log_grad_inside_sign[{{}, s, u, RIGHT, COMPLETE}], grad_sign)
	       self:logadd(self.log_grad_inside[{{}, u+1, t, LEFT, COMPLETE}],self.tmp[{{}, u}],
	 		   self.log_grad_inside_sign[{{}, u+1, t, LEFT, COMPLETE}], grad_sign)
	       self:logadd(self.log_grad_input[{{}, t, s}], self.tmp[{{}, u}],
			   self.log_grad_input_sign[{{}, t, s}], grad_sign)	    	       
	    end
	 end
      end
   end   
end

function EisnerCRF:outside_backward(input)
   -- backward step for the outside algorithm
   local batch = input:size(1)
   local n = input:size(2)
   local grad, grad_sign
   self.tmp2:resize(batch, 1)
   local l
   for k = 1, n do
      for s = 1, n do	 
	 local t = s+k
	 if t > n then break end
	 -- right incomplete
	 -- outside(right_incomplete(s,t)) gives grads to outside(right_complete(s,u))
	 -- and inside(right_complete(t,u))
	 grad = self.log_grad_outside[{{}, s, t, RIGHT, INCOMPLETE}]:add(
	       -1, self.log_outside[{{}, s, t, RIGHT, INCOMPLETE}])
	 self.tmp2[{{}, 1}]:copy(grad)	 	    	 
	 grad_sign = self.log_grad_outside_sign[{{}, s, t, RIGHT, INCOMPLETE}]
	 l = n - t + 1
	 self.tmp[{{}, {t, n}}]:copy(
	    self.log_outside[{{}, s, {}, RIGHT, COMPLETE}]:narrow(2, t, l)):add(
	    self.log_inside[{{}, t, {}, RIGHT, COMPLETE}]:narrow(2, t, l)):add(
	    self.tmp2:expand(batch, l))
	 for u = t, n do
	    self:logadd(self.log_grad_outside[{{}, s, u, RIGHT, COMPLETE}], self.tmp[{{}, u}],
			self.log_grad_outside_sign[{{}, s, u, RIGHT, COMPLETE}], grad_sign)
	    self:logadd(self.log_grad_inside[{{}, t, u, RIGHT, COMPLETE}], self.tmp[{{}, u}],
			self.log_grad_inside_sign[{{}, t, u, RIGHT, COMPLETE}], grad_sign)
	 end	 	 
	 if s > 1 then
	    -- left incomplete	    
	    -- outside(left_incomplete(s,t)) gives grads to outside(left_complete(u,t))
	    -- and inside(left_complete(u,s))	    
	    grad = self.log_grad_outside[{{}, s, t, LEFT, INCOMPLETE}]:add(
	 	  -1, self.log_outside[{{}, s, t, LEFT, INCOMPLETE}])
	    self.tmp2[{{}, 1}]:copy(grad)	 	    	    
	    grad_sign = self.log_grad_outside_sign[{{}, s, t, LEFT, INCOMPLETE}]
	    l = s
	    self.tmp[{{}, {1, s}}]:copy(
	       self.log_outside[{{}, {}, t, LEFT, COMPLETE}]:narrow(2, 1, l)):add(
	       self.log_inside[{{}, {}, s, LEFT, COMPLETE}]:narrow(2, 1, l)):add(
	       self.tmp2:expand(batch, l))	    
	    for u = 1, s do
	       self:logadd(self.log_grad_outside[{{}, u, t, LEFT, COMPLETE}], self.tmp[{{}, u}],
			   self.log_grad_outside_sign[{{}, u, t, LEFT, COMPLETE}], grad_sign)
	       self:logadd(self.log_grad_inside[{{}, u, s, LEFT, COMPLETE}], self.tmp[{{}, u}],
			   self.log_grad_inside_sign[{{}, u, s, LEFT, COMPLETE}], grad_sign)
	    end	    	    
	 end
	 -- outside(right_complete(s,t)) gives grads to
	 -- outside(right_complete(u,t)), inside(right_incomplete(u,s))
	 -- outside(right_incomplete(s,u)), inside(left_complete(t+1,u)), input(s,u)
	 -- outside(left_incomplete(s,u)), inside(left_complete(t+1,u)), input(u,s)	 
	 grad = self.log_grad_outside[{{}, s, t, RIGHT, COMPLETE}]:add(
	       -1, self.log_outside[{{}, s, t, RIGHT, COMPLETE}])
	 self.tmp2[{{}, 1}]:copy(grad)	 	    	 
	 grad_sign = self.log_grad_outside_sign[{{}, s, t, RIGHT, COMPLETE}]
	 l = s
	 self.tmp[{{}, {1, s}}]:copy(
	    self.log_outside[{{}, {}, t, RIGHT, COMPLETE}]:narrow(2, 1, l)):add(
	    self.log_inside[{{}, {}, s, RIGHT, INCOMPLETE}]:narrow(2, 1, l)):add(
	    self.tmp2:expand(batch, l))	    	 
	 for u = 1, s do
	    self:logadd(self.log_grad_outside[{{}, u, t, RIGHT, COMPLETE}], self.tmp[{{}, u}],
			self.log_grad_outside_sign[{{}, u, t, RIGHT, COMPLETE}], grad_sign)
	    self:logadd(self.log_grad_inside[{{}, u, s, RIGHT, INCOMPLETE}], self.tmp[{{}, u}],
			self.log_grad_inside_sign[{{}, u, s, RIGHT, INCOMPLETE}], grad_sign)
	 end	 
	 if t < n then
	    l = n-t	    
	    self.tmp[{{}, {t+1, n}}]:copy(
	       self.log_outside[{{}, s, {}, RIGHT, INCOMPLETE}]:narrow(2, t+1, l)):add(
	       self.log_inside[{{}, t+1, {}, LEFT, COMPLETE}]:narrow(2, t+1, l)):add(
	       input[{{}, s, {}}]:narrow(2, t+1, l)):add(
	       self.tmp2:expand(batch, l))	    
	    for u = t+1, n do
	       self:logadd(self.log_grad_outside[{{}, s, u, RIGHT, INCOMPLETE}],
			   self.tmp[{{}, u}],
	    		   self.log_grad_outside_sign[{{}, s, u, RIGHT, INCOMPLETE}], grad_sign)
	       self:logadd(self.log_grad_inside[{{}, t+1, u, LEFT, COMPLETE}],
	    		   self.tmp[{{}, u}],
	    		   self.log_grad_inside_sign[{{}, t+1, u, LEFT, COMPLETE}], grad_sign)	
	       self:logadd(self.log_grad_input[{{}, s, u}],
	    		   self.tmp[{{}, u}],
	    		   self.log_grad_input_sign[{{}, s, u}], grad_sign)
	    end	    
	    self.tmp[{{}, {t+1, n}}]:copy(
	       self.log_outside[{{}, s, {}, LEFT, INCOMPLETE}]:narrow(2, t+1, l)):add(
	       self.log_inside[{{}, t+1, {}, LEFT, COMPLETE}]:narrow(2, t+1, l)):add(
	       input[{{}, {}, s}]:narrow(2, t+1, l)):add(
	       self.tmp2:expand(batch, l))	    	    
	    for u = t+1, n do	       
	       self:logadd(self.log_grad_outside[{{}, s, u, LEFT, INCOMPLETE}],
	    		   self.tmp[{{}, u}],
	    		   self.log_grad_outside_sign[{{}, s, u, LEFT, INCOMPLETE}], grad_sign)
	       self:logadd(self.log_grad_inside[{{}, t+1, u, LEFT, COMPLETE}],
	    		   self.tmp[{{}, u}],
	    		   self.log_grad_inside_sign[{{}, t+1, u, LEFT, COMPLETE}], grad_sign)
	       self:logadd(self.log_grad_input[{{}, u, s}], self.tmp[{{}, u}],
	    		   self.log_grad_input_sign[{{}, u, s}], grad_sign)	       
	    end
	 end	 
	 if s > 1 then
	    -- outside(left_complete(s,t)) gives grads to
	    -- outside(left_complete(s,u)), inside(left_incomplete(t,u)), u = [t, n]
	    -- outside(right_incomplete(u,t)), inside(right_complete(u,s-1)), input(u,t), u = [1,s-1]
	    -- outside(left_incomplete(u,t)), inside(right_complete(u,s-1)), input(t,u), u = [1,s-1]
	    grad = self.log_grad_outside[{{}, s, t, LEFT, COMPLETE}]:add(
		  -1, self.log_outside[{{}, s, t, LEFT, COMPLETE}])
	    self.tmp2[{{}, 1}]:copy(grad)	 	    	    
	    grad_sign = self.log_grad_outside_sign[{{}, s, t, LEFT, COMPLETE}]
	    l = n - t + 1
	    self.tmp[{{}, {t, n}}]:copy(
	       self.log_outside[{{}, s, {}, LEFT, COMPLETE}]:narrow(2, t, l)):add(
	       self.log_inside[{{}, t, {}, LEFT, INCOMPLETE}]:narrow(2, t, l)):add(
	       self.tmp2:expand(batch, l))
	    for u = t, n do
	       self:logadd(self.log_grad_outside[{{}, s, u, LEFT, COMPLETE}], self.tmp[{{}, u}],
			   self.log_grad_outside_sign[{{}, s, u, LEFT, COMPLETE}], grad_sign)
	       self:logadd(self.log_grad_inside[{{}, t, u, LEFT, INCOMPLETE}], self.tmp[{{}, u}],
			   self.log_grad_inside_sign[{{}, t, u, LEFT, INCOMPLETE}], grad_sign)
	    end
	    l = s - 1
	    self.tmp[{{}, {1, s-1}}]:copy(
	       self.log_outside[{{}, {}, t, RIGHT, INCOMPLETE}]:narrow(2, 1, l)):add(
	       self.log_inside[{{}, {}, s-1, RIGHT, COMPLETE}]:narrow(2, 1, l)):add(
	       input[{{}, {}, t}]:narrow(2, 1, l)):add(self.tmp2:expand(batch, l))
	    for u = 1, s-1 do
	       self:logadd(self.log_grad_outside[{{}, u, t, RIGHT, INCOMPLETE}],
			   self.tmp[{{}, u}],
			   self.log_grad_outside_sign[{{}, u, t, RIGHT, INCOMPLETE}], grad_sign)
	       self:logadd(self.log_grad_inside[{{}, u, s-1, RIGHT, COMPLETE}],
			   self.tmp[{{}, u}],
			   self.log_grad_inside_sign[{{}, u, s-1, RIGHT, COMPLETE}], grad_sign)
	       self:logadd(self.log_grad_input[{{}, u, t}], self.tmp[{{}, u}],
			   self.log_grad_input_sign[{{}, u, t}], grad_sign)
	    end
	    self.tmp[{{}, {1, s-1}}]:copy(
	       self.log_outside[{{}, {}, t, LEFT, INCOMPLETE}]:narrow(2, 1, l)):add(
	       self.log_inside[{{}, {}, s-1, RIGHT, COMPLETE}]:narrow(2, 1, l)):add(
	       input[{{}, t, {}}]:narrow(2, 1, l)):add(self.tmp2:expand(batch, l))
	    for u = 1, s-1 do
	       self:logadd(self.log_grad_outside[{{}, u, t, LEFT, INCOMPLETE}],
			   self.tmp[{{}, u}],
			   self.log_grad_outside_sign[{{}, u, t, LEFT, INCOMPLETE}], grad_sign)
	       self:logadd(self.log_grad_inside[{{}, u, s-1, RIGHT, COMPLETE}],
			   self.tmp[{{}, u}],
			   self.log_grad_inside_sign[{{}, u, s-1, RIGHT, COMPLETE}], grad_sign)
	       self:logadd(self.log_grad_input[{{}, t, u}], self.tmp[{{}, u}],
			   self.log_grad_input_sign[{{}, t, u}], grad_sign)	       
	    end	    
	 end	 	 
      end      
   end
end

function EisnerCRF:viterbi(input)
   -- input is a (batch x n x n) of scores where
   -- input[b][s][t] is the score for connecting s ==> t
   -- chart scores are in C[b][s][t][d][c] with the following notation:
   -- b \in [1, ..., batch] = batch index 
   -- s \in [1, ..., n] = left most index in span
   -- t \in [1, ..., n] = right most index in span
   -- d \in  {1, 2} = {<==, ==>}
   -- c \in  {1, 2} = {incomplete, complete}
   local batch = input:size(1)
   local n = input:size(2)
   self.C:resize(batch, n, n, 2, 2):zero() -- initialize chart scores
   self.bp:resize(batch, n, n, 2, 2):zero() -- backpointer
   self.tmp:resize(batch, n) -- temporary scores
   local mask -- this will be used to only update batch b if the score is greater   
   for k = 1, n do
      for s = 1, n do	 
	 local t = s+k
	 if t > n then break end
	 -- initialize scores
	 self.C[{{}, s, t, LEFT, INCOMPLETE}]:fill(-math.huge)
	 self.C[{{}, s, t, RIGHT, INCOMPLETE}]:fill(-math.huge)
	 self.C[{{}, s, t, LEFT, COMPLETE}]:fill(-math.huge)
	 self.C[{{}, s, t, RIGHT, COMPLETE}]:fill(-math.huge)
	 -- create left incomplete trees	 
	 for u = s, t-1 do
	    self.tmp[{{},u}]:zero():add(self.C[{{}, s, u, RIGHT, COMPLETE}]):add(
	       self.C[{{}, u+1, t, LEFT, COMPLETE}]):add(input[{{},t,s}])	   
	    mask = self.C[{{}, s, t, LEFT, INCOMPLETE}]:lt(self.tmp[{{},u}])
	    self.C[{{}, s, t, LEFT, INCOMPLETE}][mask] = self.tmp[{{},u}][mask]
	    self.bp[{{}, s, t, LEFT, INCOMPLETE}][mask] = u
	 end
	 -- create right incomplete trees
	 for u = s, t-1 do
	    self.tmp[{{},u}]:zero():add(self.C[{{}, s, u, RIGHT, COMPLETE}]):add(
	       self.C[{{}, u+1, t, LEFT, COMPLETE}]):add(input[{{},s,t}])	   
	    mask = self.C[{{}, s, t, RIGHT, INCOMPLETE}]:lt(self.tmp[{{},u}])
	    self.C[{{}, s, t, RIGHT, INCOMPLETE}][mask] = self.tmp[{{},u}][mask]
	    self.bp[{{}, s, t, RIGHT, INCOMPLETE}][mask] = u 	       	       	    	    
	 end	 
	 -- create left complete trees
	 for u = s, t - 1 do
	    self.tmp[{{},u}]:zero():add(self.C[{{}, s, u, LEFT, COMPLETE}]):add(
	       self.C[{{}, u, t, LEFT, INCOMPLETE}])
	    mask = self.C[{{}, s, t, LEFT, COMPLETE}]:lt(self.tmp[{{},u}])
	    self.C[{{}, s, t, LEFT, COMPLETE}][mask] = self.tmp[{{},u}][mask]
	    self.bp[{{}, s, t, LEFT, COMPLETE}][mask] = u 	       	       	       	    
	 end
	 -- create right complete trees
	 for u = s+1, t do
	    self.tmp[{{},u}]:zero():add(self.C[{{}, s, u, RIGHT, INCOMPLETE}]):add(
	       self.C[{{}, u, t, RIGHT, COMPLETE}])
	    mask = self.C[{{}, s, t, RIGHT, COMPLETE}]:lt(self.tmp[{{},u}])
	    self.C[{{}, s, t, RIGHT, COMPLETE}][mask] = self.tmp[{{},u}][mask]
	    self.bp[{{}, s, t, RIGHT, COMPLETE}][mask] = u 	       	       
	 end	 	 
      end
   end
   self.heads:resize(batch, n):zero()
   for b = 1, batch do
      self:backtrack(b,1,n,2,2)
   end   
   return self.C[{{}, 1, n, RIGHT, COMPLETE}], self.heads
end

function EisnerCRF:backtrack(b, s, t, d, c)
   -- backtrack to get MAP trees
   local u = self.bp[b][s][t][d][c]
   if s == t then -- base case
      return
   elseif d == LEFT and c == INCOMPLETE then -- left incomplete
      self.heads[b][s] = t
      self:backtrack(b, s, u, RIGHT, COMPLETE)
      self:backtrack(b, u+1, t, LEFT, COMPLETE)      
   elseif d == RIGHT and c == INCOMPLETE then -- right incomplete
      self.heads[b][t] = s
      self:backtrack(b, s, u, RIGHT, COMPLETE)
      self:backtrack(b, u+1, t, LEFT, COMPLETE)      
   elseif d == LEFT and c == COMPLETE then -- left complete
      self.heads[b][u] = t
      self:backtrack(b, s, u, LEFT, COMPLETE)
      self:backtrack(b, u, t, LEFT, INCOMPLETE)      
   elseif d == RIGHT and c == COMPLETE then -- right complete
      self.heads[b][u] = s      	 
      self:backtrack(b, s, u, RIGHT, INCOMPLETE)
      self:backtrack(b, u, t, RIGHT, COMPLETE)
   end
end

