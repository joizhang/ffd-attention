from constants import *
from .celeb_df_dataset import get_celeb_df_dataloader, get_celeb_df_test_dataloader
from .deeper_forensics_dataset import get_deeper_forensics_dataloader, get_deeper_forensics_test_dataloader
from .dfdc_dataset import get_dfdc_dataloader, get_dfdc_test_dataloader
from .face_forensics_all_dataset import get_face_forensics_all_dataloader, get_face_forensics_all_test_dataloader
from .face_forensics_dataset import get_face_forensics_dataloader, get_face_forensics_test_dataloader


def get_dataloader(model_cfg, args):
    if args.prefix == FACE_FORENSICS_DF:
        train_sampler, train_loader, val_loader = get_face_forensics_dataloader(model_cfg, args, fake_type=DF)
    elif args.prefix == FACE_FORENSICS_F2F:
        train_sampler, train_loader, val_loader = get_face_forensics_dataloader(model_cfg, args, fake_type=F2F)
    elif args.prefix == FACE_FORENSICS_FSH:
        train_sampler, train_loader, val_loader = get_face_forensics_dataloader(model_cfg, args, fake_type=FSH)
    elif args.prefix == FACE_FORENSICS_FSW:
        train_sampler, train_loader, val_loader = get_face_forensics_dataloader(model_cfg, args, fake_type=FSW)
    elif args.prefix == FACE_FORENSICS_NT:
        train_sampler, train_loader, val_loader = get_face_forensics_dataloader(model_cfg, args, fake_type=NT)
    elif args.prefix == FACE_FORENSICS:
        train_sampler, train_loader, val_loader = get_face_forensics_all_dataloader(model_cfg, args)
    elif args.prefix == DEEPER_FORENSICS:
        train_sampler, train_loader, val_loader = get_deeper_forensics_dataloader(model_cfg, args)
    elif args.prefix == DFDC:
        train_sampler, train_loader, val_loader = get_dfdc_dataloader(model_cfg, args)
    else:
        train_sampler, train_loader, val_loader = get_celeb_df_dataloader(model_cfg, args)

    return train_sampler, train_loader, val_loader


def get_test_dataloader(model_cfg, args):
    if args.prefix == FACE_FORENSICS_DF:
        test_loader = get_face_forensics_test_dataloader(model_cfg, args, fake_type=DF)
    elif args.prefix == FACE_FORENSICS_F2F:
        test_loader = get_face_forensics_test_dataloader(model_cfg, args, fake_type=F2F)
    elif args.prefix == FACE_FORENSICS_FSH:
        test_loader = get_face_forensics_test_dataloader(model_cfg, args, fake_type=FSH)
    elif args.prefix == FACE_FORENSICS_FSW:
        test_loader = get_face_forensics_test_dataloader(model_cfg, args, fake_type=FSW)
    elif args.prefix == FACE_FORENSICS_NT:
        test_loader = get_face_forensics_test_dataloader(model_cfg, args, fake_type=NT)
    elif args.prefix == FACE_FORENSICS:
        test_loader = get_face_forensics_all_test_dataloader(model_cfg, args)
    elif args.prefix == DEEPER_FORENSICS:
        test_loader = get_deeper_forensics_test_dataloader(model_cfg, args)
    elif args.prefix == DFDC:
        test_loader = get_dfdc_test_dataloader(model_cfg, args)
    else:
        test_loader = get_celeb_df_test_dataloader(model_cfg, args)

    return test_loader
