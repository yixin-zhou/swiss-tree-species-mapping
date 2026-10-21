import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

class TreeSpeciesMappingMetrics:
    def __init__(self, sites_gt, sites_pred):
        assert sites_gt.shape == sites_pred.shape, (
            f"The shape of predicted composition vector is different from ground truth one. pred is {sites_pred.shape}, gt is {sites_gt.shape}."
        )

        assert sites_gt.ndim == 2, (
            f"Expected input shape [num_sites, num_species], but got {sites_gt.shape}."
        )
        
        self.sites_gt = np.asarray(sites_gt)
        self.sites_pred = np.asarray(sites_pred)

        self.sites_num = sites_gt.shape[0]
        self.species_num = sites_gt.shape[1]


    def compute_rmse(self, mode="mean_per_site"):
        allowed_modes = {
            "overall",
            "per_site",
            "mean_per_site",
            "per_species",
            "mean_per_species",
        }

        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode: '{mode}'. Expected one of {sorted(allowed_modes)}.")

        squared_error = (self.sites_pred - self.sites_gt) ** 2

        if mode == "overall":
            return np.sqrt(np.mean(squared_error))

        if mode == "per_site":
            return np.sqrt(np.mean(squared_error, axis=1))

        if mode == "mean_per_site":
            per_site_rmse = np.sqrt(np.mean(squared_error, axis=1))
            return np.mean(per_site_rmse)

        if mode == "per_species":
            return np.sqrt(np.mean(squared_error, axis=0))

        if mode == "mean_per_species":
            per_species_rmse = np.sqrt(np.mean(squared_error, axis=0))
            return np.mean(per_species_rmse)


    def compute_mae(self, mode="overall"):
        """Mean absolute error. Same aggregation axes as compute_rmse, but L1 (no
        square/sqrt) -- more robust to the extreme residuals of near-absent species,
        and the metric reported by fractional-cover mapping papers (e.g. Klehr et
        al. 2025, RSE)."""
        allowed_modes = {
            "overall",
            "per_site",
            "mean_per_site",
            "per_species",
            "mean_per_species",
        }
        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode: '{mode}'. Expected one of {sorted(allowed_modes)}.")

        abs_error = np.abs(self.sites_pred - self.sites_gt)

        if mode == "overall":
            return np.mean(abs_error)

        if mode == "per_site":
            return np.mean(abs_error, axis=1)

        if mode == "mean_per_site":
            return np.mean(np.mean(abs_error, axis=1))

        if mode == "per_species":
            return np.mean(abs_error, axis=0)

        if mode == "mean_per_species":
            return np.mean(np.mean(abs_error, axis=0))


    def compute_nrmse(self):
        per_species_rmse = self.compute_rmse(mode="per_species")
        gt_std_per_species = np.std(self.sites_gt, axis=0)

        nrmse = np.full(self.species_num, np.nan)

        valid = gt_std_per_species != 0
        nrmse[valid] = per_species_rmse[valid] / gt_std_per_species[valid]

        return nrmse

    
    def compute_r2(self, mode="overall"):
        allowed_modes = ["overall", "per_species", "macro"]

        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode: {mode}. Expected one of {allowed_modes}.")

        if mode == "overall":
            gt = self.sites_gt.reshape(-1)
            pred = self.sites_pred.reshape(-1)

            ss_res = np.sum((gt - pred) ** 2)
            ss_tot = np.sum((gt - np.mean(gt)) ** 2)

            if ss_tot == 0:
                return np.nan

            return 1 - ss_res / ss_tot

        if mode == "per_species":
            ss_res = np.sum((self.sites_gt - self.sites_pred) ** 2, axis=0)
            gt_mean_per_species = np.mean(self.sites_gt, axis=0)
            ss_tot = np.sum((self.sites_gt - gt_mean_per_species) ** 2, axis=0)

            r2 = np.full(self.species_num, np.nan)

            valid = ss_tot != 0
            r2[valid] = 1 - ss_res[valid] / ss_tot[valid]

            return r2

        if mode == "macro":
            per_species_r2 = self.compute_r2(mode="per_species")
            return np.nanmean(per_species_r2)


    def dominant_species_metrics(
        self,
        top_n=(1, 2, 3),
        other_index=-1,
        exclude_other_dominant=True,
    ):

        if other_index is not None:
            if not isinstance(other_index, int):
                raise ValueError(f"other_index must be an integer or None, got {other_index}.")

            if not -self.species_num <= other_index < self.species_num:
                raise ValueError(
                    f"other_index={other_index} is out of range for "
                    f"species_num={self.species_num}."
                )

            other_index = other_index % self.species_num

        if isinstance(top_n, int):
            top_n = (top_n,)

        for n in top_n:
            if not isinstance(n, int) or n < 1:
                raise ValueError(f"Each n in top_n must be a positive integer, got {n}.")
            if n > self.species_num:
                raise ValueError(
                    f"n cannot be larger than number of species/classes. "
                    f"Got n={n}, species_num={self.species_num}."
                )

        true_dominant = np.argmax(self.sites_gt, axis=1)
        pred_dominant = np.argmax(self.sites_pred, axis=1)

        valid_mask = np.ones(self.sites_num, dtype=bool)

        if exclude_other_dominant:
            if other_index is None:
                raise ValueError(
                    "other_index must not be None when exclude_other_dominant=True."
                )
            valid_mask &= true_dominant != other_index

        if not np.any(valid_mask):
            result = {
                "n_sites": 0,
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "macro_precision": np.nan,
                "macro_recall": np.nan,
                "confusion_matrix": None,
                "labels": None,
            }

            for n in top_n:
                result[f"top{n}_accuracy"] = np.nan

            return result

        y_true = true_dominant[valid_mask]
        y_pred = pred_dominant[valid_mask]
        pred_vectors = self.sites_pred[valid_mask]

        labels = list(range(self.species_num))
        if exclude_other_dominant and other_index is not None:
            labels = [label for label in labels if label != other_index]

        result = {
            "n_sites": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
            "macro_f1": f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            ),
            "macro_precision": precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            "macro_recall": recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
            "labels": labels,
        }

        for n in top_n:
            pred_top_n = np.argsort(pred_vectors, axis=1)[:, -n:]
            correct = np.any(pred_top_n == y_true[:, None], axis=1)
            result[f"top{n}_accuracy"] = np.mean(correct)

        return result
    
    
    def compute_bray_curtis(self, mode="mean_per_site"):
        allowed_modes = {"per_site", "mean_per_site"}

        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode: '{mode}'. Expected one of {sorted(allowed_modes)}.")

        numerator = np.sum(np.abs(self.sites_pred - self.sites_gt), axis=1)
        denominator = np.sum(self.sites_pred + self.sites_gt, axis=1)

        bc = np.full(self.sites_num, np.nan)
        valid = denominator != 0
        bc[valid] = numerator[valid] / denominator[valid]

        if mode == "per_site":
            return bc

        return np.nanmean(bc)
    
    def get_all_metrics(
        self,
        dominant_top_n=(1, 2, 3),
        other_index=-1,
        exclude_other_dominant=True,
    ):
        metrics = {
            "rmse_overall": self.compute_rmse(mode="overall"),
            "rmse_mean_per_site": self.compute_rmse(mode="mean_per_site"),
            "rmse_per_site": self.compute_rmse(mode="per_site"),
            "rmse_per_species": self.compute_rmse(mode="per_species"),
            "rmse_mean_per_species": self.compute_rmse(mode="mean_per_species"),
            "mae_overall": self.compute_mae(mode="overall"),
            "mae_per_species": self.compute_mae(mode="per_species"),
            "mae_mean_per_species": self.compute_mae(mode="mean_per_species"),
            "nrmse_per_species": self.compute_nrmse(),
            "r2_overall": self.compute_r2(mode="overall"),
            "r2_per_species": self.compute_r2(mode="per_species"),
            "r2_macro": self.compute_r2(mode="macro"),
            "bray_curtis_mean_per_site": self.compute_bray_curtis(mode="mean_per_site"),
            "bray_curtis_per_site": self.compute_bray_curtis(mode="per_site"),
            "dominant_species": self.dominant_species_metrics(
                top_n=dominant_top_n,
                other_index=other_index,
                exclude_other_dominant=exclude_other_dominant,
            ),
        }

        return metrics
