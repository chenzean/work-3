import torch.optim as optim


def get_optimizer(args, parameters):
    if args.optimizer == 'Adam':
        return optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay,
                          betas=(0.9, 0.999), amsgrad=args.amsgrad, eps=args.eps)

    elif args.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        return optim.SGD(parameters, lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(args.optim.optimizer))
