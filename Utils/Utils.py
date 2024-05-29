import torch as t
import os
def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	# ret += (model.usrStruct + model.itmStruct)
	return ret

# def calcReward(bprLoss, keepRate):
# 	# return t.where(bprLoss >= threshold, 1.0, xi)
# 	_, posLocs = t.topk(bprLoss, int(bprLoss.shape[0] * (1 - keepRate)))
# 	ones = t.ones_like(bprLoss).cuda()
# 	reward = t.minimum(bprLoss, ones * (0.5 - 1e-6))
# 	pckBprLoss = bprLoss[posLocs]
# 	ones = t.ones_like(pckBprLoss).cuda()
# 	reward[posLocs] = t.minimum(t.maximum(pckBprLoss, ones * (0.5 + 1e-6)), ones)
# 	return reward

def calcReward(bprLossDiff, keepRate):
	_, posLocs = t.topk(bprLossDiff, int(bprLossDiff.shape[0] * (1 - keepRate)))
	reward = t.zeros_like(bprLossDiff).cuda()
	reward[posLocs] = 1.0
	return reward

def calcGradNorm(model):
	ret = 0
	for p in model.parameters():
		if p.grad is not None:
			ret += p.grad.data.norm(2).square()
	ret = (ret ** 0.5)
	ret.detach()
	return ret

def getFileName(save,epoch,args):

	file = f"autocf-{args.data}--{epoch}.pth.tar"
	return os.path.join(save,'model',file)

def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.makedirs(path)
		os.mkdir(os.path.join(path, 'model'))

	print('Experiment dir : {}'.format(path))
	if scripts_to_save is not None:
		os.mkdir(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)
