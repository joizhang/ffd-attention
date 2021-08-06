from constants import FACE_FORENSICS_DF, FACE_FORENSICS_FSH, CELEB_DF, DFDC
from training.datasets.celeb_df_dataset import get_celeb_df_dataloader, get_celeb_df_test_dataloader
from training.datasets.dfdc_dataset import get_dfdc_dataloader, get_dfdc_test_dataloader
from training.datasets.dffd_dataset import get_dffd_dataloader
from training.datasets.face_forensics_dataset import get_face_forensics_dataloader, get_face_forensics_test_dataloader


def get_dataloader(model_cfg, args):
    if args.prefix == FACE_FORENSICS_DF:
        train_sampler, train_loader, val_loader = get_face_forensics_dataloader(model_cfg, args, fake_type='Deepfakes')
    elif args.prefix == FACE_FORENSICS_FSH:
        train_sampler, train_loader, val_loader = get_face_forensics_dataloader(model_cfg, args, fake_type='FaceShifter')
    elif args.prefix == CELEB_DF:
        train_sampler, train_loader, val_loader = get_celeb_df_dataloader(model_cfg, args)
    elif args.prefix == DFDC:
        train_sampler, train_loader, val_loader = get_dfdc_dataloader(model_cfg, args)
    else:
        train_sampler, train_loader = get_dffd_dataloader(model_cfg, args, 'train', num_workers=0)
        val_loader = get_dffd_dataloader(model_cfg, args, 'validation', shuffle=False)

    return train_sampler, train_loader, val_loader


def get_test_dataloader(model_cfg, args):
    if args.prefix == FACE_FORENSICS_DF:
        test_loader = get_face_forensics_test_dataloader(model_cfg, args, fake_type='Deepfakes')
    elif args.prefix == FACE_FORENSICS_FSH:
        test_loader = get_face_forensics_test_dataloader(model_cfg, args, fake_type='FaceShifter')
    elif args.prefix == CELEB_DF:
        test_loader = get_celeb_df_test_dataloader(model_cfg, args)
    elif args.prefix == DFDC:
        test_loader = get_dfdc_test_dataloader(model_cfg, args)
    else:
        test_loader = get_dffd_dataloader(model_cfg, args, 'test', shuffle=False)

    return test_loader
