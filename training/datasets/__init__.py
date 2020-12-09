from preprocessing.constants import FACE_FORENSICS_DF, FACE_FORENSICS_FSH, CELEB_DF
from training.datasets.celeb_df_dataset import get_celeb_df_dataloader, get_celeb_df_test_dataloader
from training.datasets.dffd_dataset import get_dffd_dataloader
from training.datasets.face_forensics_dataset import get_face_forensics_dataloader, get_face_forensics_test_dataloader


def get_dataloader(model, args):
    if args.prefix == FACE_FORENSICS_DF:
        train_sampler, train_loader, val_loader = get_face_forensics_dataloader(model, args, fake_type='Deepfakes')
    elif args.prefix == FACE_FORENSICS_FSH:
        train_sampler, train_loader, val_loader = get_face_forensics_dataloader(model, args, fake_type='FaceShifter')
    elif args.prefix == CELEB_DF:
        train_sampler, train_loader, val_loader = get_celeb_df_dataloader(model, args)
    else:
        train_sampler, train_loader = get_dffd_dataloader(model, args, 'train', num_workers=0)
        val_loader = get_dffd_dataloader(model, args, 'validation', shuffle=False)

    return train_sampler, train_loader, val_loader


def get_test_dataloader(model, args):
    if args.prefix == FACE_FORENSICS_DF:
        test_loader = get_face_forensics_test_dataloader(model, args, fake_type='Deepfakes')
    elif args.prefix == FACE_FORENSICS_FSH:
        test_loader = get_face_forensics_test_dataloader(model, args, fake_type='FaceShifter')
    elif args.prefix == CELEB_DF:
        test_loader = get_celeb_df_test_dataloader(model, args)
    else:
        test_loader = get_dffd_dataloader(model, args, 'test', shuffle=False)

    return test_loader
