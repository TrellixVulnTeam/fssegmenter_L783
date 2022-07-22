import segm.utils.torch as ptu
from torchvision import transforms

from segm.data import ImagenetDataset
from segm.data import ADE20KSegmentation
from segm.data import PascalContextDataset
from segm.data import Pascal5iDataset
from segm.data import CityscapesDataset
from segm.data import Loader


def create_dataset(dataset_kwargs):
    dataset_kwargs = dataset_kwargs.copy()
    dataset_name = dataset_kwargs.pop("dataset")
    batch_size = dataset_kwargs.pop("batch_size")
    num_workers = dataset_kwargs.pop("num_workers")
    split = dataset_kwargs.pop("split")
    fold = dataset_kwargs.pop("fold")
    shot = dataset_kwargs.pop("shot")

    # load dataset_name
    if dataset_name == "imagenet":
        dataset_kwargs.pop("patch_size")
        dataset = ImagenetDataset(split=split, **dataset_kwargs)
    elif dataset_name == "ade20k":
        dataset = ADE20KSegmentation(split=split, **dataset_kwargs)
    elif dataset_name == "pascal_context":
        dataset = PascalContextDataset(split=split, **dataset_kwargs)
    elif dataset_name == "pascal5i":
        datapath = r'/home/prlab/wxl/dataset/dir/pcontext/VOCdevkit'
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im_size = dataset_kwargs.pop("image_size")
        transform = transforms.Compose([transforms.Resize(size=(im_size, im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(img_mean, img_std)])
        use_original_imgsize = False
        dataset = Pascal5iDataset(datapath,fold,transform,split,shot,use_original_imgsize)
    elif dataset_name == "cityscapes":
        dataset = CityscapesDataset(split=split, **dataset_kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

    dataset = Loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=ptu.distributed,
        split=split,
    )
    return dataset
