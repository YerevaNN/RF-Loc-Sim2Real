import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler


class DatamoduleBase(pl.LightningDataModule):
    
    collate_fn = None
    
    def __init__(self, batch_size: int, num_workers: int, drop_last: bool, multi_gpu: bool = False, *args, **kwargs):
        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        
        self.train_set_field = None
        self.val_set_field = None
        self.test_set_field = None
        self.args = args
        self.kwargs = kwargs
        
        self.multi_gpu = multi_gpu
        if multi_gpu:
            self.prepare_data()
    
    def train_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self.train_set_field) if self.multi_gpu else None
        return DataLoader(
            self.train_set_field, batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=sampler, shuffle=None if self.multi_gpu else True, collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
    
    def val_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self.val_set_field, shuffle=False) if self.multi_gpu else None
        return DataLoader(
            self.val_set_field, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler,
            collate_fn=self.collate_fn, drop_last=self.drop_last
        )
    
    def test_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self.test_set_field, shuffle=False) if self.multi_gpu else None
        return DataLoader(
            self.test_set_field, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler,
            collate_fn=self.collate_fn, drop_last=self.drop_last
        )
    
    @property
    def train_set(self):
        return self.train_set_field
    
    @property
    def test_set(self):
        return self.test_set_field
    
    @property
    def val_set(self):
        return self.val_set_field
