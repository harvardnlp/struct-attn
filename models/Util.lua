require "sys"
local Util, parent = torch.class("Util")

local function isnan(i) return (i == i and i or -math.huge) end
local function isinf(i) return (i > -1e10 and i or -1e10) end

function Util.logbmm2(res, tensor1, tensor2, t1)
    -- performs batch matrix multiplication in log space
    local batch = tensor2:size(1)
    local dim2 = tensor2:size()
    local dim1 = tensor1:size()

    v1 = tensor1:view(batch, dim1[2], 1, dim2[2]):expand(batch, dim1[2], dim2[3], dim2[2])
    v2 = tensor2:view(batch, 1, dim2[3], dim2[2]):expand(batch, dim1[2], dim2[3], dim2[2])
    if t1 then
        t1:copy(v1):add(v2)
    else
        t1 = v2:clone():add(v1)
    end
    if dim2[2] ~= 1 then
        res2 = res:view(batch, dim1[2], dim2[3], 1)
        Util.logsum2(res2, nil, t1, 4, nil, true)
    else
        res:copy(t1:squeeze())
    end
end

function Util.logbmm(res, resSign, tensor1, tensor2, tensor1Sign, tensor2Sign, 
        t1, t2, t1_start, t2_start)
    -- performs batch matrix multiplication in log space
    local batch, dim1, dim2 = tensor2:size(1), tensor1:size(), tensor2:size()
    local exp = dim2[3]

    v1 = tensor1:view(batch, dim1[2], 1, dim2[2]):expand(batch, dim1[2], dim2[3], dim2[2])
    v2 = tensor2:view(batch, 1, dim2[3], dim2[2]):expand(batch, dim1[2], dim2[3], dim2[2])

    u1 = tensor1Sign:view(batch, dim1[2], 1, dim2[2]):expand(batch, dim1[2], dim2[3], dim2[2])
    u2 = tensor2Sign:view(batch, 1, dim2[3], dim2[2]):expand(batch, dim1[2], dim2[3], dim2[2])

    if t1 then
        t1:narrow(3, t1_start or 1, t1:size(3)-((t1_start or 1) -1)):copy(v1):add(v2)
    else
        t1 = v2:clone():add(v1)
    end

    if t2 then
        t2:narrow(3, t2_start or 1, t1:size(3)-((t2_start or 1) -1)):copy(u1):cmul(u2)
    else
        t2 = u2:clone():cmul(u1)
    end

    if t1:size(3) ~= 1 then
        res2 = res:view(batch, dim1[2], 1, dim2[3])
        resSign2 = resSign:view(batch, dim1[2], 1, dim2[3])
        Util.logsum2(res2, resSign2, t1, 3, t2, nil)
    else
        res:copy(t1:squeeze())
        resSign:copy(t2:squeeze())

        Util.fixnan(res)
    end
end


function Util.logadd(result, sign, tensor1, tensor2, tensor1Sign, tensor2Sign, max, ge)
    if max then
        ge = max:copy(tensor1):add(-1, tensor2):ge(0):typeAs(tensor1)
        max:copy(tensor1):cmax(tensor2)
    else
        max = torch.cmax(tensor1, tensor2)
        ge = torch.add(tensor1, -1, tensor2):ge(0):typeAs(tensor1)
    end
    result:cmin(tensor1, tensor2):add(-1, max)
    result:exp():cmul(tensor1Sign):cmul(tensor2Sign):log1p():add(max)
    sign:copy(tensor1Sign):cmul(ge)

    ge:add(-1):mul(-1)
    ge:cmul(tensor2Sign)
    sign:add(ge)
    Util.fixnan(result)
end

function Util.logsum2(result, sign, tensor, dim, tensorSign, dontchecknan)
    local max = tensor:max(dim)
    tensor:add(-1, max:expandAs(tensor)):exp()
    if tensorSign then
        tensor:cmul(tensorSign)
    end
    local sumExp = tensor:sum(dim)
    if sign then
        sign:copy(torch.sign(sumExp))
    end
    result:copy(max):add(sumExp:abs():log())
    if dontchecknan then
    else
        Util.fixnan(result)
    end
end


function Util.logsum(tensor, dim, tensorSign, dontchecknan)
    tensorSign = tensorSign or torch.ones(tensor:size()):typeAs(tensor)
    local max = tensor:max(dim)
    local diff = torch.add(tensor, -1, max:expandAs(tensor))
    local sumExp = torch.exp(diff):cmul(tensorSign):sum(dim)
    local sign = torch.sign(sumExp)
    local sum = torch.add(max, sumExp:abs():log())
    if dontchecknan then
    else
        sum:apply(isnan)
    end
    return sum, sign
end

function Util.logsumNumber(tensor, tensorSign)
    local sum = tensor:view(tensor:size(1), tensor:size(2) *  tensor:size(3) * tensor:size(4))
    local sign = tensorSign:view(tensor:size(1), tensor:size(2) * tensor:size(3) * tensor:size(4))
    local sum2 = torch.Tensor(tensor:size(1)):typeAs(sum)
    local sign2 = torch.Tensor(tensor:size(1)):typeAs(sign)

    Util.logsum2(sum2, sign2, sum:clone(), 2, sign, true)
    sum2 = sum2:view(sum2:size(1))
    sign2 = sign2:view(sum2:size(1))

    Util.fixnan(sum2)
    return sum2, sign2
end

function Util.fixnan(v)
    if v.THCUDAMOD then
        v.THCUDAMOD.FixNaN(v:cdata())
    else
        v:apply(isnan)
    end
end
