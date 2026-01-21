"""
SST-Merge: Data-Free Version

LoRAアダプターの構造のみからFIMを近似計算し、
学習データなしでSST-Mergeを実行
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FIMCalculatorDataFree:
    """
    データフリーFIM計算
    
    LoRAの低ランク構造 ΔW = B × A を利用して
    FIM ≈ ||ΔW||² を近似計算
    """
    
    def __init__(self, regularization: float = 1e-6):
        self.regularization = regularization
    
    def compute_fim_from_lora(self, adapter_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        LoRAアダプターからFIMを近似計算（データ不要）
        
        Args:
            adapter_dict: LoRAアダプターの重み辞書
        
        Returns:
            fim_diag: 対角FIM (1次元テンソル)
        """
        fim_components = []
        
        # LoRA A/B ペアを探索
        lora_a_keys = [k for k in adapter_dict.keys() if 'lora_A' in k]
        
        logger.info(f"Found {len(lora_a_keys)} LoRA layers")
        
        for lora_a_key in lora_a_keys:
            lora_b_key = lora_a_key.replace('lora_A', 'lora_B')
            
            if lora_b_key not in adapter_dict:
                logger.warning(f"Missing lora_B for {lora_a_key}")
                continue
            
            lora_A = adapter_dict[lora_a_key]  # (r, d_in)
            lora_B = adapter_dict[lora_b_key]  # (d_out, r)
            
            # ΔW = B @ A
            delta_w = torch.matmul(lora_B, lora_A)  # (d_out, d_in)
            
            # FIM ≈ ||ΔW||² (element-wise)
            fim = delta_w.pow(2).flatten()
            
            fim_components.append(fim)
        
        if not fim_components:
            raise ValueError("No LoRA parameters found in adapter_dict")
        
        # 全パラメータを結合
        fim_diag = torch.cat(fim_components)
        
        # 正則化
        fim_diag = fim_diag + self.regularization
        
        logger.info(f"FIM computed (data-free): {len(fim_diag)} parameters, "
                   f"mean={fim_diag.mean():.6f}, std={fim_diag.std():.6f}")
        
        return fim_diag


class GEVPSolver:
    """
    Generalized Eigenvalue Problem Solver
    
    既存実装と同じ（sst_merge.pyから流用）
    """
    
    def __init__(self, regularization: float = 1e-6):
        self.regularization = regularization
    
    def solve_gevp_diagonal(
        self,
        F_harm: torch.Tensor,
        F_benign: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        対角FIM用のGEVPを解く
        
        λ_i = F_harm[i] / F_benign[i]
        
        Args:
            F_harm: Safety/Harm FIM (対角)
            F_benign: Utility/Benign FIM (対角)
            
        Returns:
            eigenvalues: 全パラメータの固有値 (λ)
            sorted_indices: 降順ソートのインデックス
        """
        # 正則化を追加してゼロ除算を防ぐ
        F_benign_reg = F_benign + self.regularization
        F_harm_reg = F_harm + self.regularization
        
        # 固有値 λ = F_harm / F_benign
        eigenvalues = F_harm_reg / F_benign_reg
        
        # 降順ソート
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        
        logger.info(f"GEVP solved: {len(eigenvalues)} eigenvalues")
        logger.info(f"  λ range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
        logger.info(f"  λ mean: {eigenvalues.mean():.6f}, median: {eigenvalues.median():.6f}")
        
        return eigenvalues, sorted_indices
    
    def compute_safety_mask(
        self,
        eigenvalues: torch.Tensor,
        top_k_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        固有値λに基づくSafety適用マスクを計算
        
        高いλ → Safety追加しても安全（マスク値大）
        低いλ → Utilityに影響するため控えめに（マスク値小）
        
        Args:
            eigenvalues: GEVP固有値
            top_k_ratio: Top-k選択比率 (None=ソフトマスク, 0.3=上位30%のみ)
        
        Returns:
            mask: Safety適用マスク [0, 1]
        """
        if top_k_ratio is not None:
            # Hard Top-k マスク
            k = int(len(eigenvalues) * top_k_ratio)
            threshold = torch.topk(eigenvalues, k).values.min()
            mask = (eigenvalues >= threshold).float()
            logger.info(f"Hard mask: Top {top_k_ratio*100:.1f}% ({k} params)")
        else:
            # Soft マスク (シグモイド正規化)
            normalized = (eigenvalues - eigenvalues.mean()) / (eigenvalues.std() + 1e-8)
            mask = torch.sigmoid(normalized)
            logger.info(f"Soft mask: mean={mask.mean():.4f}")
        
        return mask


class SSTMergeDataFree:
    """
    データフリーSST-Merge
    
    LoRAアダプターのみでSST-Mergeを実行
    """
    
    def __init__(
        self,
        safety_weight: float = 0.5,
        use_gevp: bool = True,
        regularization: float = 1e-6,
        top_k_ratio: Optional[float] = None,
        use_layerwise: bool = False
    ):
        """
        Args:
            safety_weight: 基本Safety重み (α)
            use_gevp: GEVP-basedマスクを使用
            regularization: FIM正則化項
            top_k_ratio: Top-k選択比率 (None=ソフトマスク)
            use_layerwise: Layerwise重み調整（データフリー版では未実装、将来拡張用）
        """
        self.safety_weight = safety_weight
        self.use_gevp = use_gevp
        self.regularization = regularization
        self.top_k_ratio = top_k_ratio
        self.use_layerwise = use_layerwise  # 将来の拡張用
        
        self.fim_calc = FIMCalculatorDataFree(regularization)
        self.gevp_solver = GEVPSolver(regularization)
    
    def merge(
        self,
        utility_adapter: Dict[str, torch.Tensor],
        safety_adapter: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        データフリーSST-Mergeを実行
        
        Args:
            utility_adapter: Utility LoRAアダプター
            safety_adapter: Safety LoRAアダプター
        
        Returns:
            merged_adapter: マージされたアダプター
        """
        logger.info("="*60)
        logger.info("Starting Data-Free SST-Merge")
        logger.info(f"  Safety weight (α): {self.safety_weight}")
        logger.info(f"  Use GEVP: {self.use_gevp}")
        logger.info(f"  Top-k ratio: {self.top_k_ratio}")
        logger.info("="*60)
        
        if self.use_gevp:
            return self._merge_with_gevp(utility_adapter, safety_adapter)
        else:
            return self._merge_simple(utility_adapter, safety_adapter)
    
    def _merge_with_gevp(
        self,
        utility_adapter: Dict[str, torch.Tensor],
        safety_adapter: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """GEVP-based merge（データフリー版）"""
        
        # Step 1: Utility FIM計算（データ不要）
        logger.info("\nStep 1: Computing Utility FIM (data-free)...")
        F_benign = self.fim_calc.compute_fim_from_lora(utility_adapter)
        
        # Step 2: Safety FIM計算（データ不要）
        logger.info("\nStep 2: Computing Safety FIM (data-free)...")
        F_harm = self.fim_calc.compute_fim_from_lora(safety_adapter)
        
        # Step 3: GEVP解法
        logger.info("\nStep 3: Solving GEVP...")
        eigenvalues, _ = self.gevp_solver.solve_gevp_diagonal(F_harm, F_benign)
        
        # Step 4: Safetyマスク計算
        logger.info("\nStep 4: Computing Safety mask...")
        mask = self.gevp_solver.compute_safety_mask(eigenvalues, self.top_k_ratio)
        
        # Step 5: マージ実行
        logger.info("\nStep 5: Merging adapters...")
        merged_adapter = {}
        idx = 0
        
        for key in utility_adapter.keys():
            u_param = utility_adapter[key]
            s_param = safety_adapter[key]
            
            param_numel = u_param.numel()
            param_mask = mask[idx:idx+param_numel].reshape(u_param.shape)
            
            # Merge: Utility + α × mask × Safety
            merged_adapter[key] = u_param + self.safety_weight * param_mask * s_param
            
            idx += param_numel
        
        logger.info(f"Merge completed: {len(merged_adapter)} parameters")
        
        return merged_adapter
    
    def _merge_simple(
        self,
        utility_adapter: Dict[str, torch.Tensor],
        safety_adapter: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """シンプルなTask Arithmetic風マージ"""
        
        merged_adapter = {}
        for key in utility_adapter.keys():
            merged_adapter[key] = (
                utility_adapter[key] + 
                self.safety_weight * safety_adapter[key]
            )
        
        logger.info(f"Simple merge completed: {len(merged_adapter)} parameters")
        
        return merged_adapter


def save_merged_adapter(
    merged_adapter: Dict[str, torch.Tensor], 
    output_path: str,
    source_adapter_path: str = None
):
    """
    マージされたアダプターを保存
    
    Args:
        merged_adapter: マージされたアダプター
        output_path: 出力パス
        source_adapter_path: 元のアダプターパス（設定ファイルのコピー元）
    """
    import shutil
    import json
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # アダプター重みを保存
    torch.save(merged_adapter, output_path / "adapter_model.bin")
    
    # adapter_config.jsonをコピー（PEFTが認識するため必須）
    if source_adapter_path:
        source_path = Path(source_adapter_path)
        config_file = source_path / "adapter_config.json"
        
        if config_file.exists():
            shutil.copy(config_file, output_path / "adapter_config.json")
            logger.info(f"  Copied adapter_config.json from {source_adapter_path}")
        else:
            logger.warning(f"  adapter_config.json not found in {source_adapter_path}")
        
        # README.mdもコピー（オプション）
        readme_file = source_path / "README.md"
        if readme_file.exists():
            shutil.copy(readme_file, output_path / "README.md")
    
    logger.info(f"Merged adapter saved to: {output_path}")


if __name__ == "__main__":
    # Quick test
    print("Data-Free SST-Merge module loaded successfully")
