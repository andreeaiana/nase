from typing import Any, List, Dict, Optional, Union

import hydra
from lightning import LightningModule
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.utils import to_dense_batch
from torchmetrics.retrieval import RetrievalMRR
from torchmetrics import MeanMetric

from nase.data.components.rec_batch import RecommendationBatch
from nase.models.components.click_predictor import DotProduct
from nase.models.components.denoising_autoencoder_loss import DenoisingAutoEncoderLoss
from nase.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class NASE(LightningModule):
    def __init__(
            self,
            language_model: DictConfig,
            tokenizer: DictConfig,
            tgt_languages: List[str],
            val_batch_size: int,
            optimizer: DictConfig,
            scheduler: Optional[DictConfig] = None,
            ) -> None:
        super().__init__()
        
        self.save_hyperparameters(logger=False)

        model = hydra.utils.instantiate(self.hparams.language_model)
        tokenizer = hydra.utils.instantiate(self.hparams.tokenizer)
        self.loss_fct = DenoisingAutoEncoderLoss(
                model=model,
                tokenizer=tokenizer,
                decoder_name_or_path=self.hparams.language_model,
                tie_encoder_decoder=True
                )

        self.click_predictor = DotProduct()
        
        self.train_loss = MeanMetric()      
        self.mrr = nn.ModuleList([RetrievalMRR() for _ in range(len(self.hparams.tgt_languages))]) 

    def forward(self, batch) -> torch.Tensor:
        loss = self.loss_fct(batch)
        return loss

    def training_step(self, batch: dict, batch_idx: int):
        loss = self(batch)
        
        self.train_loss(loss)
        self.log(
                "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True 
                )

        return loss   
    
    def validation_step(self, batch: RecommendationBatch, batch_idx: int, dataloader_idx: int) -> None:
        # encode history
        hist_news_vector = self.loss_fct.encoder(**batch['x_hist']['text']).pooler_output
        hist_news_vector_agg, mask_hist = to_dense_batch(hist_news_vector, batch['batch_hist'])
        
        # encode candidates
        cand_news_vector = self.loss_fct.encoder(**batch['x_cand']['text']).pooler_output
        cand_news_vector_agg, _ = to_dense_batch(cand_news_vector, batch['batch_cand'])
        
        # aggregate embeddings of clicked news
        hist_size = torch.tensor(
                [torch.where(mask_hist[i])[0].shape[0] for i in range(mask_hist.shape[0])],
                device=self.device
                )
        user_vector = torch.div(
                hist_news_vector_agg.sum(dim=1),
                hist_size.unsqueeze(dim=-1)
                )

        # click scores
        scores = self.click_predictor(
                user_vector.unsqueeze(dim=1),
                cand_news_vector_agg.permute(0, 2, 1)
                )

        y_true, mask_cand = to_dense_batch(batch["labels"], batch["batch_cand"])
        
        # model outputs for metric computation
        preds = torch.cat(
                [scores[n][mask_cand[n]] for n in range(mask_cand.shape[0])],
                dim=0
                )
        targets = torch.cat(
                [y_true[n][mask_cand[n]] for n in range(mask_cand.shape[0])],
                dim=0
                )
        cand_news_size = torch.tensor(
            [torch.where(mask_cand[n])[0].shape[0] for n in range(mask_cand.shape[0])]
        )
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)
        
        self.mrr[dataloader_idx].update(preds, targets, indexes)
        self.log(
            f"val/mrr_{dataloader_idx}", 
            self.mrr[dataloader_idx], 
            on_step=False, 
            on_epoch=True, 
            prog_bar=True, 
            logger=True, 
            batch_size=self.hparams.val_batch_size
        )

    def on_validation_epoch_end(self):
        # compute average MRR across all dataloaders
        avg_mrr = torch.stack(
                [self.mrr[idx].compute() for idx in range(len(self.mrr))]
                ).mean()

        # log average MRR
        self.log(
                "val/avg_mrr",
                avg_mrr,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True
                )
    
    def test_step(self, batch: RecommendationBatch, batch_idx: int) -> None:
        pass

    def on_test_epoch_end(self):
        pass 

    def configure_scheduler(
         self, 
         optimizer: Optimizer, 
         scheduler_cfg: DictConfig
         ) -> dict[str, Union[str, int, LambdaLR]]:
         
         if hasattr(scheduler_cfg, "num_warmup_steps") and isinstance(
            scheduler_cfg.num_warmup_steps, (int, float)
            ):
             scheduler_cfg.num_warmup_steps *= self.trainer.max_steps
             scheduler_cfg.num_training_steps = self.trainer.max_steps
             log.info(
                     f"Warm up for {scheduler_cfg.num_warmup_steps} of {self.trainer.max_steps}"
                     )

         scheduler = hydra.utils.instantiate(
                 scheduler_cfg,
                 optimizer,
         )
         
         return {
                "scheduler": scheduler, 
                "interval": "step", 
                "frequency": 1
                }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Prepares optimizer and scheduler."""
        if weight_decay := getattr(self.hparams.optimizer, "weight_decay", None):
            param_optimizer = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            parameters = [
                    {
                        "params": [
                            p
                            for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)
                            ],
                        "weight_decay": weight_decay,
                        },
                    {
                        "params": [
                            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                            ],
                        "weight_decay": 0.0,
                        },
                    ]
        else:
            parameters = self.parameters()

        optimizer = hydra.utils.instantiate(self.hparams.optimizer, parameters) 
        if scheduler_cfg := getattr(self.hparams, "scheduler"):
            scheduler = self.configure_scheduler(optimizer, scheduler_cfg)
            return {
                    "optimizer": optimizer,
                    "scheduler": scheduler
                    } 
        return {"optimizer": optimizer}
