import torch


def restore_snapshot(net, snapshot ):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    print("Checkpoint Load Compelete")

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore1(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore1(net, checkpoint)

    return net


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    count_all,count_same1,count_same2 = 0, 0,0
    for k in net_state_dict:
        count_all += 1
        if k.split('.')[0] == 'resnet_features':
            if k[16:] in loaded_dict and net_state_dict[k].size() == loaded_dict[k[16:]].size():
                new_loaded_dict[k] = loaded_dict[k[16:]]
                count_same1 += 1
            elif k[16:] in loaded_dict and net_state_dict[k].size() != loaded_dict[k[16:]].size():
                count_same2 += 1
            else:
                print("跳过{0}的参数加载".format(k))
    print('总参数{}个,相同参数{}个,大小不同{}个'.format(count_all,count_same1,count_same2))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

def forgiving_state_restore1(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    count_all,count_same1,count_same2 = 0, 0,0
    for k in net_state_dict:
        count_all += 1
        if k.split('.')[0] == 'resnet_features':
            if k[16:] in loaded_dict and net_state_dict[k].size() == loaded_dict[k[16:]].size():
                new_loaded_dict[k] = loaded_dict[k[16:]]
                count_same1 += 1
            elif k[16:] in loaded_dict and net_state_dict[k].size() != loaded_dict[k[16:]].size():
                count_same2 += 1
            else:
                print("跳过{0}的参数加载".format(k))
    print('总参数{}个,相同参数{}个,大小不同{}个'.format(count_all,count_same1,count_same2))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net