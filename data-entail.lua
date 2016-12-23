--
-- Manages encoder/decoder data matrices.
--

local data = torch.class("data")

function data:__init(opt, data_file)
   local f = hdf5.open(data_file, 'r')
   
   self.source  = f:read('source'):all()
   self.target  = f:read('target'):all()   
   self.target_l = f:read('target_l'):all() --max target length each batch
   self.source_l = f:read('source_l'):all()
   self.label = f:read('label'):all()
   self.batch_l = f:read('batch_l'):all()
   self.batch_idx = f:read('batch_idx'):all()
   self.target_size = f:read('target_size'):all()[1]
   self.source_size = f:read('source_size'):all()[1]
   self.label_size = f:read('label_size'):all()[1]
   self.length = self.batch_l:size(1)
   self.seq_length = self.target:size(2) 
   self.batches = {}
   for i = 1, self.length do
      local source_i =  self.source:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
				  1, self.source_l[i])
      local target_i = self.target:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
				       1, self.target_l[i])
      local label_i = self.label:sub(self.batch_idx[i], self.batch_idx[i] + self.batch_l[i]-1)
      table.insert(self.batches,  {target_i, source_i, self.batch_l[i], self.target_l[i],
				   self.source_l[i], label_i})
   end
end

function data:size()
   return self.length
end

function data.__index(self, idx)
   if type(idx) == "string" then
      return data[idx]
   else
      local target = self.batches[idx][1]
      local source = self.batches[idx][2]      
      local batch_l = self.batches[idx][3]
      local target_l = self.batches[idx][4]
      local source_l = self.batches[idx][5]
      local label = self.batches[idx][6]
      if opt.gpuid >= 0 then --if multi-gpu, source lives in gpuid1, rest on gpuid2
	 source = source:cuda()
	 target = target:cuda()
	 label = label:cuda()
      end
      return {target, source, batch_l, target_l, source_l, label}
   end
end

return data
