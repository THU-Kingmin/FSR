import os

def get_model_path(args,epoch):
    model_path = os.path.join(args.ckp_dir,'FSR_{}_P{}_X{}_E{}.pth.tar'.\
        format(args.SRmodule,args.patch,args.scale,epoch))
    return model_path