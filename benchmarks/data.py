"""Dataset do benchmark: carga, splits estratificados e ruído AWGN."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkData:
    """Conjunto (X, y) homogêneo usado pelo benchmark.

    X tem forma (N, *input_shape) — espectrograma (T, F) para modelos neurais,
    achatado para (N, T*F) pelos modelos clássicos (SVM/RF) no runner.
    y é 1D com rótulos {0=real/bonafide, 1=fake/spoof}.
    """

    X: np.ndarray
    y: np.ndarray
    name: str = "dataset"
    metadata: Dict[str, Any] | None = None
    # P0 — fonte/gerador por amostra (alinhado a X/y). Usado para splits
    # disjuntos por grupo e para o protocolo cross-generator. None quando o
    # dataset não carrega proveniência.
    groups: np.ndarray | None = None
    # Tier `large` — falante por amostra (alinhado a X/y). Usado para split
    # disjunto por falante e protocolo holdout-speaker (usuários não vistos).
    # None quando o dataset não carrega identificação de falante.
    speakers: np.ndarray | None = None
    # Hierarquia de proveniencia para auditoria, bootstrap e relatorios.
    utterances: np.ndarray | None = None
    texts: np.ndarray | None = None
    generators: np.ndarray | None = None
    cluster_ids: np.ndarray | None = None
    speaker_known: np.ndarray | None = None
    generator_known: np.ndarray | None = None
    sample_paths: np.ndarray | None = None
    # Índices das partições fornecidas no NPZ. Preservá-los impede que runners
    # diferentes reconstruam conjuntos de treino/validação/teste distintos.
    predefined_split_indices: Dict[str, np.ndarray] | None = None
    # Índices efetivamente usados pela chamada mais recente de
    # ``stratified_split``. Diferentemente de ``predefined_split_indices``,
    # estes refletem também protocolos por grupo, falante ou holdout.
    last_split_indices: Dict[str, np.ndarray] | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def synthetic(
        cls, n: int = 360, shape: Tuple[int, int] = (32, 16), seed: int = 42
    ) -> "BenchmarkData":
        """Dataset sintético separável (verificação do harness, sem áudio).

        A classe fake recebe um deslocamento de média → linearmente separável
        em poucas épocas. NÃO substitui dados reais; serve para testar o pipeline.
        """
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n,) + tuple(shape)).astype("float32")
        y = rng.integers(0, 2, n).astype("int64")
        X[y == 1] += 0.6
        return cls(X=X, y=y, name="synthetic", metadata={"source": "synthetic"})

    @classmethod
    def from_npz(cls, path: str) -> "BenchmarkData":
        """Carrega de um .npz e preserva os indices train/val/test predefinidos.
        A concatenacao cria uma visao comum sem alterar o teste congelado.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {path}")
        data = np.load(p, allow_pickle=False)
        xs, ys = [], []
        used_keys: list[tuple[str, str]] = []
        split_indices: Dict[str, np.ndarray] | None = None
        has_predefined = all(
            key in data
            for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
        )
        if has_predefined:
            pairs = (
                ("X_train", "y_train"),
                ("X_val", "y_val"),
                ("X_test", "y_test"),
            )
        elif "X" in data and "y" in data:
            pairs = (("X", "y"),)
        else:
            pairs = tuple(
                (xk, yk)
                for xk, yk in (
                    ("X_train", "y_train"),
                    ("X_val", "y_val"),
                    ("X_test", "y_test"),
                )
                if xk in data and yk in data
            )
        offset = 0
        if has_predefined:
            split_indices = {}
        for xk, yk in pairs:
            if xk in data and yk in data:
                x_part = np.asarray(data[xk], dtype="float32")
                y_part = np.asarray(data[yk])
                xs.append(x_part)
                ys.append(y_part)
                used_keys.append((xk, yk))
                if split_indices is not None:
                    split_name = xk.removeprefix("X_")
                    split_indices[split_name] = np.arange(
                        offset, offset + len(y_part), dtype="int64"
                    )
                    offset += len(y_part)
        if not xs:
            raise ValueError(
                f"{path}: esperado X_train/y_train (ou X/y) no .npz; "
                f"chaves encontradas: {list(data.keys())}"
            )
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        # Normaliza rótulos para {0,1} (1D, inteiro)
        y = np.asarray(y)
        if y.ndim > 1 and y.shape[-1] > 1:
            y = np.argmax(y, axis=-1)
        y = y.ravel().astype("int64")
        metadata: Dict[str, Any] = {"source": str(p)}
        if "metadata_json" in data:
            try:
                raw_meta = data["metadata_json"]
                if hasattr(raw_meta, "item"):
                    raw_meta = raw_meta.item()
                metadata.update(json.loads(str(raw_meta)))
                metadata["npz_path"] = str(p)
            except Exception:
                metadata["metadata_parse_error"] = True

        # P0 — proveniência por amostra (fonte/gerador). Preferimos uma chave
        # explícita `groups` no .npz; senão derivamos do nome de arquivo nos
        # `paths` do metadata (mesma ordem de concatenação: train→val→test→X).
        groups = cls._extract_groups(data, metadata, used_keys, n=len(y))
        speakers = cls._extract_speakers(data, metadata, used_keys, n=len(y))
        utterances = cls._extract_aligned(data, "utterance_ids", len(y), str)
        texts = cls._extract_aligned(data, "text_ids", len(y), str)
        generators = cls._extract_aligned(data, "generator_ids", len(y), str)
        cluster_ids = cls._extract_aligned(data, "cluster_ids", len(y), str)
        speaker_known = cls._extract_aligned(data, "speaker_known", len(y), bool)
        generator_known = cls._extract_aligned(data, "generator_known", len(y), bool)
        sample_paths = cls._extract_aligned(data, "sample_paths", len(y), str)
        loaded = cls(
            X=X,
            y=y,
            name=p.stem,
            metadata=metadata,
            groups=groups,
            speakers=speakers,
            utterances=utterances,
            texts=texts,
            generators=generators,
            cluster_ids=cluster_ids,
            speaker_known=speaker_known,
            generator_known=generator_known,
            sample_paths=sample_paths,
            predefined_split_indices=split_indices,
        )
        loaded.validate()
        return loaded

    @staticmethod
    def _derive_group(path: str) -> str:
        """Extrai o identificador de fonte/gerador do nome do arquivo.

        Os nomes seguem `<fonte>_NNNNN.wav` (ex.: brspeech, cvpt, fkvoice). Não
        há ID de falante embutido, então o grupo mais fino disponível é a
        FONTE/GERADOR — suficiente para o reteste cross-generator (XTTS=fkvoice).
        """
        base = str(path).replace("\\", "/").rsplit("/", 1)[-1].lower()
        m = re.match(r"([a-z]+)", base)
        return m.group(1) if m else "unknown"

    @staticmethod
    def _extract_aligned(
        data: Any,
        key: str,
        n: int,
        dtype: type = str,
    ) -> np.ndarray | None:
        """Return an explicit per-sample vector only when alignment is exact."""
        if key not in getattr(data, "files", []):
            return None
        values = np.asarray(data[key]).ravel()
        if len(values) != n:
            return None
        return values.astype(dtype)

    @classmethod
    def _extract_groups(
        cls,
        data: Any,
        metadata: Dict[str, Any],
        used_keys: list[tuple[str, str]],
        n: int,
    ) -> np.ndarray | None:
        """Constrói o array de grupos alinhado a X/y, ou None se indisponível."""
        if "groups" in getattr(data, "files", []):
            g = np.asarray(data["groups"]).astype(str).ravel()
            return g if len(g) == n else None

        splits = (metadata or {}).get("splits") or {}
        key_to_split = {"X_train": "train", "X_val": "val", "X_test": "test"}
        groups: list[str] = []
        for xk, _yk in used_keys:
            split_name = key_to_split.get(xk)
            paths = (splits.get(split_name) or {}).get("paths") if split_name else None
            if not paths:
                return None  # sem proveniência completa → não arrisca desalinhar
            groups.extend(cls._derive_group(p) for p in paths)
        if len(groups) != n:
            return None
        return np.asarray(groups, dtype=object).astype(str)

    @classmethod
    def _extract_speakers(
        cls,
        data: Any,
        metadata: Dict[str, Any],
        used_keys: list[tuple[str, str]],
        n: int,
    ) -> np.ndarray | None:
        """Array de falantes alinhado a X/y (chave `speaker_ids` no .npz), ou None.

        Preferimos a chave explícita `speaker_ids`; senão derivamos do
        `speaker_manifest` a partir dos `paths` do metadata (mesma ordem de
        concatenação train→val→test→X).
        """
        if "speaker_ids" in getattr(data, "files", []):
            s = np.asarray(data["speaker_ids"]).astype(str).ravel()
            return s if len(s) == n else None

        splits = (metadata or {}).get("splits") or {}
        key_to_split = {"X_train": "train", "X_val": "val", "X_test": "test"}
        try:
            from app.domain.dataset_metadata.speaker_manifest import speaker_for_path
        except Exception:
            return None
        speakers: list[str] = []
        for xk, _yk in used_keys:
            split_name = key_to_split.get(xk)
            paths = (splits.get(split_name) or {}).get("paths") if split_name else None
            if not paths:
                return None
            speakers.extend(speaker_for_path(p) for p in paths)
        if len(speakers) != n:
            return None
        return np.asarray(speakers, dtype=object).astype(str)

    def validate(self, min_per_class: int = 2) -> None:
        """Valida sanidade básica antes de treinar/avaliar."""
        X = np.asarray(self.X)
        y = np.asarray(self.y)
        if X.ndim < 2:
            raise ValueError(
                f"{self.name}: X deve ter batch + features, shape={X.shape}"
            )
        if len(X) != len(y):
            raise ValueError(f"{self.name}: len(X)={len(X)} difere de len(y)={len(y)}")
        if not np.isfinite(X).all():
            raise ValueError(f"{self.name}: X contém NaN ou Inf")
        aligned = {
            "groups": self.groups,
            "speakers": self.speakers,
            "utterances": self.utterances,
            "texts": self.texts,
            "generators": self.generators,
            "cluster_ids": self.cluster_ids,
            "speaker_known": self.speaker_known,
            "generator_known": self.generator_known,
            "sample_paths": self.sample_paths,
        }
        for key, values in aligned.items():
            if values is not None and len(values) != len(y):
                raise ValueError(
                    f"{self.name}: {key} desalinhado ({len(values)} != {len(y)})"
                )
        if not np.isfinite(y).all():
            raise ValueError(f"{self.name}: y contém NaN ou Inf")
        labels, counts = np.unique(y.ravel().astype("int64"), return_counts=True)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError(
                f"{self.name}: labels esperados {{0,1}}, encontrados {labels.tolist()}"
            )
        too_small = {
            int(k): int(v) for k, v in zip(labels, counts) if v < min_per_class
        }
        if too_small:
            raise ValueError(
                f"{self.name}: classes com amostras insuficientes para split: {too_small}"
            )

    def prepare_for_architecture(self, architecture: str) -> "BenchmarkData":
        """Retorna uma visão de X compatível com o contrato da arquitetura."""
        prepared, actual_type = prepare_input_for_architecture(self.X, architecture)
        meta = dict(self.metadata or {})
        meta.update(
            {
                "architecture": architecture,
                "input_type": actual_type,
                "original_shape": list(np.asarray(self.X).shape[1:]),
                "prepared_shape": list(np.asarray(prepared).shape[1:]),
            }
        )
        return BenchmarkData(
            X=np.asarray(prepared, dtype="float32"),
            y=np.asarray(self.y, dtype="int64"),
            name=self.name,
            metadata=meta,
            groups=self.groups,
            speakers=self.speakers,
            utterances=self.utterances,
            texts=self.texts,
            generators=self.generators,
            cluster_ids=self.cluster_ids,
            speaker_known=self.speaker_known,
            generator_known=self.generator_known,
            sample_paths=self.sample_paths,
            predefined_split_indices=self.predefined_split_indices,
        )

    def stratified_split(
        self,
        seed: int = 42,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        group_split: bool = False,
        holdout_generator: str | None = None,
        speaker_split: bool = False,
        holdout_speaker: str | None = None,
        preserve_predefined: bool = True,
    ):
        """Divisão 70/15/15. Suporta cinco modos (precedência nesta ordem):

        - **holdout_speaker**: protocolo usuário não visto — segura um falante
          fora do treino e o usa (só ele + reais reservados) como teste.
        - **holdout_generator**: protocolo cross-generator — segura um gerador
          fora do treino e o usa como teste.
        - **speaker_split**: mantém FALANTE disjunto entre train/val/test.
        - **group_split**: mantém fonte/gerador DISJUNTO entre train/val/test.
        - **estratificada** (default): preserva a proporção de classes.
        """
        if holdout_speaker is not None and self.speakers is None:
            raise ValueError("holdout_speaker exige speaker_ids explicitos")
        if speaker_split and self.speakers is None:
            raise ValueError("speaker_split exige speaker_ids explicitos")
        if (holdout_speaker is not None or speaker_split) and (
            self.speaker_known is None or not np.all(self.speaker_known)
        ):
            raise ValueError(
                "Protocolo por falante recusado: cobertura speaker_known incompleta"
            )
        if holdout_generator is not None:
            self._require_generator_holdout(holdout_generator)
        if group_split and self.groups is None:
            raise ValueError("group_split exige grupos de proveniencia")

        reparticiona = (
            holdout_speaker is not None
            or holdout_generator is not None
            or speaker_split
            or group_split
        )
        # Um dataset com particao predefinida traz um teste SELADO e ja disjunto.
        # Reparticionar por cima dele descarta esse selo e, no caso do
        # `speaker_split`, mantem o locutor disjunto mas devolve o texto para as
        # tres particoes — vazamento de conteudo travestido de protocolo. Exigir
        # `preserve_predefined=False` torna a escolha explicita de quem chama.
        if reparticiona and preserve_predefined and self.predefined_split_indices:
            raise ValueError(
                "Protocolo alternativo recusado: o dataset traz particao "
                "predefinida (teste selado, disjunto por locutor e texto). "
                "Reparticionar descartaria essa garantia. Passe "
                "preserve_predefined=False para assumir a troca de forma "
                "explicita, ciente de que o resultado deixa de ser comparavel "
                "com o teste selado."
            )

        if holdout_speaker is not None and self.speakers is not None:
            return self._cross_generator_split(
                holdout_speaker, seed, val_frac, groups=self.speakers
            )
        if holdout_generator is not None:
            return self._cross_generator_split(
                holdout_generator, seed, val_frac, groups=self.generators
            )
        if speaker_split and self.speakers is not None:
            return self._grouped_split(
                seed,
                val_frac,
                test_frac,
                groups=self.speakers,
                relaxed={"texts": ()},
            )
        if group_split and self.groups is not None:
            return self._grouped_split(
                seed,
                val_frac,
                test_frac,
                relaxed={"speakers": (), "texts": ()},
            )
        if preserve_predefined and self.predefined_split_indices:
            required = {"train", "val", "test"}
            if required.issubset(self.predefined_split_indices):
                tr = self.predefined_split_indices["train"]
                va = self.predefined_split_indices["val"]
                te = self.predefined_split_indices["test"]
                self._validate_partition_indices(tr, va, te)
                self.last_split_indices = {
                    "train": np.asarray(tr, dtype="int64"),
                    "val": np.asarray(va, dtype="int64"),
                    "test": np.asarray(te, dtype="int64"),
                }
                return (
                    self.X[tr],
                    self.y[tr],
                    self.X[va],
                    self.y[va],
                    self.X[te],
                    self.y[te],
                )
        try:
            from sklearn.model_selection import train_test_split

            idx = np.arange(len(self.y))
            train_idx, temp_idx = train_test_split(
                idx,
                test_size=val_frac + test_frac,
                stratify=self.y,
                random_state=seed,
            )
            rel_test = test_frac / (val_frac + test_frac)
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=rel_test,
                stratify=self.y[temp_idx],
                random_state=seed,
            )
        except Exception as exc:
            raise RuntimeError(
                "Split estratificado inviavel; aumente o dataset ou corrija os rotulos"
            ) from exc
        self._validate_partition_indices(train_idx, val_idx, test_idx)
        self.last_split_indices = {
            "train": np.asarray(train_idx, dtype="int64"),
            "val": np.asarray(val_idx, dtype="int64"),
            "test": np.asarray(test_idx, dtype="int64"),
        }
        return (
            self.X[train_idx],
            self.y[train_idx],
            self.X[val_idx],
            self.y[val_idx],
            self.X[test_idx],
            self.y[test_idx],
        )

    def _select(self, idx: np.ndarray):
        idx = np.asarray(idx, dtype=int)
        return self.X[idx], self.y[idx]

    #: Dimensoes cuja repeticao entre particoes e vazamento. Cada uma so e
    #: verificada quando o dataset carrega o vetor correspondente.
    DISJOINT_DIMENSIONS = (
        ("sample_paths", "amostra"),
        ("utterances", "enunciado"),
        ("speakers", "locutor"),
        ("texts", "texto"),
    )

    def _validate_partition_indices(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        *,
        relaxed: Dict[str, Tuple[str, ...]] | None = None,
    ) -> None:
        """Recusa partições vazias, sobrepostas ou sem as duas classes.

        Verifica também que nenhuma **amostra, enunciado, locutor ou texto** se
        repete entre treino, validação e teste — a especificação do dataset
        (docs/data/dataset-protocol.md). Sem isto, um protocolo que reparticiona
        pode desfazer em silêncio a disjunção dupla que o artefato garante: a
        checagem de índices sozinha só prova que as partições não compartilham
        LINHAS, não que não compartilham conteúdo.

        `relaxed` mapeia dimensão -> pares de partição em que o protocolo em curso
        não pode garantir disjunção por construção. Tupla vazia relaxa todos os
        pares; `("trainxval",)` relaxa só aquele. Um holdout de falante, por
        exemplo, garante locutor inédito **no teste**, mas treino e validação
        compartilham locutores de propósito — são o conjunto "visto". Toda
        relaxação é registrada, nunca silenciosa.
        """
        relaxed = relaxed or {}
        partitions = {
            "train": np.asarray(train_idx, dtype="int64"),
            "val": np.asarray(val_idx, dtype="int64"),
            "test": np.asarray(test_idx, dtype="int64"),
        }
        for name, indices in partitions.items():
            if len(indices) == 0:
                raise ValueError(f"Particao {name} vazia")
            labels = set(np.asarray(self.y)[indices].astype(int).tolist())
            if labels != {0, 1}:
                raise ValueError(
                    f"Particao {name} deve conter real e fake; "
                    f"rotulos={sorted(labels)}"
                )
        names = tuple(partitions)
        for pos, left in enumerate(names):
            for right in names[pos + 1 :]:
                overlap = np.intersect1d(partitions[left], partitions[right])
                if len(overlap):
                    raise ValueError(f"Particoes {left}/{right} sobrepostas")

        for attribute, rotulo in self.DISJOINT_DIMENSIONS:
            values = getattr(self, attribute, None)
            if values is None or len(values) != len(self.y):
                continue
            liberados = relaxed.get(attribute)
            values = np.asarray(values)
            by_split = {
                name: set(values[idx].tolist()) for name, idx in partitions.items()
            }
            for pos, left in enumerate(names):
                for right in names[pos + 1 :]:
                    par = f"{left}x{right}"
                    if liberados is not None and (not liberados or par in liberados):
                        logger.warning(
                            "Protocolo nao garante disjuncao de %s entre %s; "
                            "metricas nao medem generalizacao nessa dimensao.",
                            rotulo,
                            par,
                        )
                        continue
                    shared = by_split[left] & by_split[right]
                    if shared:
                        exemplo = sorted(str(v) for v in shared)[:3]
                        raise ValueError(
                            f"Vazamento de {rotulo}: {len(shared)} valor(es) "
                            f"compartilhado(s) entre {left} e {right}; "
                            f"exemplos={exemplo}. O dataset exige treino, "
                            f"validacao e teste sem repeticao (ver "
                            f"docs/data/dataset-protocol.md)."
                        )

    def _require_generator_holdout(self, holdout_generator: str) -> None:
        """Recusa o protocolo cross-generator quando ele nao e aplicavel.

        O holdout era resolvido contra `groups` (a FONTE), nao contra
        `generators`. Isso funcionava por coincidencia em acervos onde a fonte e
        o gerador eram a mesma coisa; num corpus pareado, em que as duas classes
        compartilham a fonte de proposito, o protocolo silenciosamente nao
        encontrava o gerador.
        """
        if self.generators is None:
            raise ValueError(
                "holdout_generator exige `generator_ids` por amostra no .npz"
            )
        disponiveis = sorted(set(np.asarray(self.generators).astype(str).tolist()))
        alvo = holdout_generator.lower()
        if not any(g.lower() == alvo for g in disponiveis):
            raise ValueError(
                f"Gerador holdout inexistente: {holdout_generator}; "
                f"disponiveis={disponiveis}"
            )
        sinteticos = [g for g in disponiveis if g.lower() not in {"bonafide", "real"}]
        if len(sinteticos) < 2:
            raise ValueError(
                "Protocolo cross-generator inaplicavel: o corpus tem um unico "
                f"gerador sintetico ({sinteticos}). Segura-lo esvazia a classe "
                "falsa. Generalizacao para geradores nao vistos exige um "
                "conjunto externo com outro gerador (ver "
                "docs/data/dataset-protocol.md, secao 8)."
            )

    def _grouped_split(
        self,
        seed: int,
        val_frac: float,
        test_frac: float,
        groups: np.ndarray | None = None,
        relaxed: Dict[str, Tuple[str, ...]] | None = None,
    ):
        """Split disjunto por grupo via StratifiedGroupKFold (anti-vazamento).

        Mantém cada grupo (fonte/gerador, ou falante quando `groups=self.speakers`)
        inteiramente em um único conjunto. Como pode haver poucos grupos
        correlacionados à classe, isto pode desbalancear classes — é o trade-off
        honesto para eliminar vazamento.
        """
        from sklearn.model_selection import StratifiedGroupKFold

        groups = np.asarray(self.groups if groups is None else groups)
        idx = np.arange(len(self.y))
        n_groups = len(np.unique(groups))
        if n_groups < 3:
            raise ValueError(
                "Split por grupo exige pelo menos 3 grupos explicitos; "
                f"encontrados={n_groups}"
            )
        # nº de folds limitado pelo nº de grupos; teste = 1 fold.
        n_splits = max(2, min(round(1.0 / max(test_frac, 1e-6)), n_groups))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        trainval_idx, test_idx = next(sgkf.split(idx, self.y, groups))
        # Val a partir do trainval, ainda disjunto por grupo quando possível.
        g_tv = groups[trainval_idx]
        if len(np.unique(g_tv)) >= 2:
            rel_val = val_frac / (1.0 - test_frac)
            inner_splits = max(
                2, min(round(1.0 / max(rel_val, 1e-6)), len(np.unique(g_tv)))
            )
            sgkf2 = StratifiedGroupKFold(
                n_splits=inner_splits, shuffle=True, random_state=seed
            )
            tr_rel, val_rel = next(
                sgkf2.split(trainval_idx, self.y[trainval_idx], g_tv)
            )
            train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[val_rel]
        else:
            raise ValueError("Split por grupo nao consegue criar validacao disjunta")
        self._validate_partition_indices(train_idx, val_idx, test_idx, relaxed=relaxed)
        self.last_split_indices = {
            "train": np.asarray(train_idx, dtype="int64"),
            "val": np.asarray(val_idx, dtype="int64"),
            "test": np.asarray(test_idx, dtype="int64"),
        }
        Xtr, ytr = self._select(train_idx)
        Xv, yv = self._select(val_idx)
        Xte, yte = self._select(test_idx)
        return Xtr, ytr, Xv, yv, Xte, yte

    def _cross_generator_split(
        self,
        holdout_generator: str,
        seed: int,
        val_frac: float,
        groups: np.ndarray | None = None,
    ):
        """Protocolo cross-generator / holdout-speaker: treina SEM o item segurado,
        testa NELE.

        Teste = todas as amostras do gerador/falante segurado + reais não vistos
        no treino (para manter ambas as classes no teste). Train/val saem do
        restante, estratificados por classe. Com `groups=self.speakers` vira o
        protocolo de usuário não visto.
        """
        from sklearn.model_selection import train_test_split

        eh_falante = groups is self.speakers
        groups = np.asarray(self.groups if groups is None else groups)
        held = np.char.lower(groups.astype(str)) == holdout_generator.lower()
        if not held.any():
            available = sorted(set(groups.astype(str).tolist()))
            rotulo = "Falante" if eh_falante else "Gerador"
            raise ValueError(
                f"{rotulo} holdout inexistente: {holdout_generator}; "
                f"disponiveis={available}"
            )

        idx = np.arange(len(self.y))
        held_idx = idx[held]
        rest_idx = idx[~held]
        y_rest = self.y[rest_idx]

        # Quando o item segurado ja traz as DUAS classes — o caso do corpus
        # pareado, em que o mesmo locutor aparece como bonafide e como clone —
        # completar com reais do restante so desbalancearia o teste (medido:
        # 30 reais para 10 falsas). Nesse caso o holdout basta por si.
        held_labels = set(np.asarray(self.y)[held_idx].astype(int).tolist())
        if held_labels == {0, 1}:
            real_test_idx = np.asarray([], dtype=held_idx.dtype)
            test_idx = held_idx
        else:
            # Item de classe unica (ex.: um gerador sintetico): reservamos reais
            # ineditos no treino para o teste ter as duas classes.
            real_rest = rest_idx[y_rest == 0]
            rng = np.random.default_rng(seed)
            real_rest = rng.permutation(real_rest)
            n_real_test = min(len(real_rest), max(1, len(held_idx)))
            real_test_idx = real_rest[:n_real_test]
            test_idx = np.concatenate([held_idx, real_test_idx])

        trainval_idx = np.setdiff1d(rest_idx, real_test_idx, assume_unique=False)
        y_tv = self.y[trainval_idx]
        try:
            tr_idx, val_idx = train_test_split(
                trainval_idx,
                test_size=val_frac,
                stratify=y_tv,
                random_state=seed,
            )
        except Exception as exc:
            raise RuntimeError(
                "Holdout nao permite train/val estratificados com ambas as classes"
            ) from exc

        # Um holdout segura UMA dimensao e deixa as outras cruzarem de proposito:
        # segurando o locutor, o texto reaparece nas tres particoes; segurando o
        # gerador, locutor e texto reaparecem. Declarar isso mantem a checagem
        # honesta em vez de desliga-la por inteiro.
        relaxed = (
            {"texts": (), "speakers": ("trainxval",)}
            if eh_falante
            else {"speakers": (), "texts": ()}
        )
        self._validate_partition_indices(tr_idx, val_idx, test_idx, relaxed=relaxed)
        self.last_split_indices = {
            "train": np.asarray(tr_idx, dtype="int64"),
            "val": np.asarray(val_idx, dtype="int64"),
            "test": np.asarray(test_idx, dtype="int64"),
        }
        Xtr, ytr = self._select(tr_idx)
        Xv, yv = self._select(val_idx)
        Xte, yte = self._select(test_idx)
        return Xtr, ytr, Xv, yv, Xte, yte

    @staticmethod
    def add_awgn(X: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
        """Adiciona AWGN com SNR realizado igual ao alvo, por amostra.

        Esta função opera no domínio recebido. No protocolo científico do
        benchmark, deve receber exclusivamente a forma de onda canônica; a
        conversão para log-Mel ou descritores tabulares ocorre depois.
        """
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype="float32")
        flat = X.reshape(len(X), -1)
        sig_power = np.mean(flat**2, axis=1, keepdims=True)
        snr_lin = 10.0 ** (float(snr_db) / 10.0)

        unit_noise = rng.standard_normal(flat.shape).astype("float32")
        unit_power = np.mean(unit_noise**2, axis=1, keepdims=True)
        target_power = sig_power / max(snr_lin, 1e-12)
        scale = np.sqrt(target_power / np.maximum(unit_power, 1e-12))
        noise = unit_noise * scale
        return (flat + noise).reshape(X.shape).astype("float32")

    @staticmethod
    def balanced_snr_assignments(
        n_samples: int,
        snr_levels_db: list[int] | tuple[int, ...],
        seed: int = 0,
    ) -> np.ndarray:
        """Distribui níveis de SNR globalmente, com diferença máxima de um."""

        levels = np.asarray(list(snr_levels_db), dtype="float32")
        if levels.size == 0:
            raise ValueError("snr_levels_db não pode ser vazio")
        assigned = np.resize(levels, int(n_samples)).copy()
        np.random.default_rng(seed).shuffle(assigned)
        return assigned

    @staticmethod
    def add_awgn_assigned(
        X: np.ndarray,
        assigned_snr_db: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """Aplica o nível previamente atribuído a cada forma de onda."""

        X = np.asarray(X, dtype="float32")
        assigned = np.asarray(assigned_snr_db, dtype="float32").ravel()
        if len(X) != len(assigned):
            raise ValueError("assigned_snr_db deve ter uma entrada por amostra")
        noisy = np.empty_like(X, dtype="float32")
        for offset, snr in enumerate(np.unique(assigned).tolist()):
            mask = assigned == snr
            noisy[mask] = BenchmarkData.add_awgn(
                X[mask], float(snr), seed=seed + 1009 * (offset + 1)
            )
        return noisy

    @staticmethod
    def add_awgn_mixed(
        X: np.ndarray,
        snr_levels_db: list[int] | tuple[int, ...],
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gera uma cópia ruidosa por amostra com SNRs balanceados."""

        X = np.asarray(X, dtype="float32")
        assigned = BenchmarkData.balanced_snr_assignments(
            len(X), snr_levels_db, seed=seed
        )
        noisy = BenchmarkData.add_awgn_assigned(X, assigned, seed=seed)
        return noisy, assigned


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_classical_arch(name: str) -> bool:
    return _slug(name) in {"svm", "randomforest"}


def _architecture_input_contract(architecture: str) -> tuple[str, Dict[str, Any]]:
    if _is_classical_arch(architecture):
        return "tabular", {}
    try:
        from app.domain.models.architectures.factory import (
            architecture_factory_registry,
        )
        from app.domain.models.architectures.registry import normalize_arch_name

        normalized = normalize_arch_name(architecture)
        spec = architecture_factory_registry.get_architecture_info(normalized)
        if spec:
            req = dict(spec.input_requirements or {})
            return str(req.get("input_type", "spectrogram")), req
    except Exception:
        pass
    return "spectrogram", {}


def prepare_input_for_architecture(
    X: np.ndarray,
    architecture: str,
    *,
    crop_strategy: str = "center",
    seed: Optional[int] = None,
) -> tuple[np.ndarray, str]:
    """Aplica o frontend de ``architecture`` a um lote ainda no domínio bruto.

    É a fronteira usada pelo novo protocolo AWGN: primeiro o ruído é adicionado
    à forma de onda; somente depois esta função produz raw-audio normalizado,
    log-Mel ou o vetor tabular de 63 descritores.
    """
    input_type, requirements = _architecture_input_contract(architecture)
    if _is_classical_arch(architecture):
        prepared = _to_tabular_features(X)
        actual_type = (
            "tabular_audio_features"
            if _looks_like_raw_audio(X)
            else "tabular_flattened"
        )
    elif input_type == "raw_audio":
        prepared = _to_raw_audio(
            X,
            requirements,
            crop_strategy=crop_strategy,
            seed=seed,
        )
        actual_type = "raw_audio"
    elif input_type == "spectrogram":
        prepared = _to_spectrogram(X, requirements)
        actual_type = "spectrogram"
    else:
        prepared = np.asarray(X, dtype="float32")
        actual_type = input_type or "unchanged"
    return np.asarray(prepared, dtype="float32"), actual_type


def looks_like_raw_audio(X: np.ndarray) -> bool:
    """API pública para validar se um lote contém formas de onda."""
    return _looks_like_raw_audio(X)


def _fit_length(flat: np.ndarray, target_len: int) -> np.ndarray:
    # Fonte única treino<->inferência: app/domain/features/benchmark_frontend.
    from app.domain.features.benchmark_frontend import fit_length_tile

    return fit_length_tile(flat, target_len)


def _resize_axis(X: np.ndarray, target: int, axis: int) -> np.ndarray:
    current = X.shape[axis]
    if current == target:
        return X
    if current > target:
        slices = [slice(None)] * X.ndim
        slices[axis] = slice(0, target)
        return X[tuple(slices)]
    pad_width = [(0, 0)] * X.ndim
    pad_width[axis] = (0, target - current)
    return np.pad(X, pad_width, mode="edge")


def _normalize_per_sample(X: np.ndarray) -> np.ndarray:
    from app.domain.features.benchmark_frontend import normalize_per_sample

    return normalize_per_sample(X)


def _looks_like_raw_audio(X: np.ndarray) -> bool:
    arr = np.asarray(X)
    if arr.ndim == 2:
        sample_len = arr.shape[1]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        sample_len = arr.shape[1]
    else:
        return False
    return sample_len >= 1000


def _audio_flat(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype="float32").reshape(len(X), -1)


def _to_tabular_features(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype="float32")
    if not _looks_like_raw_audio(arr):
        return arr

    # Fonte única do vetor de 63 descritores (11 temporais + 26 MFCC +
    # 26 RASTA-PLP): app/domain/features/benchmark_frontend — a MESMA função
    # usada pela inferência do app (paridade por construção).
    from app.domain.features.benchmark_frontend import tabular_features_batch

    return tabular_features_batch(arr)


def _rasta_plp_stats(flat: np.ndarray, n_plp: int = 13) -> np.ndarray:
    from app.domain.features.benchmark_frontend import _rasta_plp_stats as _impl

    return _impl(flat, n_plp=n_plp)


def _to_raw_audio(
    X: np.ndarray,
    requirements: Dict[str, Any],
    *,
    crop_strategy: str = "center",
    seed: Optional[int] = None,
) -> np.ndarray:
    arr = np.asarray(X, dtype="float32")
    flat = arr.reshape(len(arr), -1)
    min_len = int(
        requirements.get("min_sequence_length")
        or requirements.get("sample_rate", 16000)
    )
    # `min_sequence_length` is a lower bound. Architectures may request an
    # explicit `target_sequence_length` when full clips are too memory-heavy.
    # Without an explicit target, preserve long raw clips instead of truncating
    # them to the minimum.
    target_len = int(
        requirements.get("sequence_length")
        or requirements.get("target_sequence_length")
        or max(flat.shape[1], min_len)
    )
    from app.domain.features.benchmark_frontend import raw_audio_batch

    return raw_audio_batch(
        flat,
        target_len=max(1, target_len),
        crop_strategy=crop_strategy,
        seed=seed,
    )


def _raw_audio_to_logmel(X: np.ndarray, requirements: Dict[str, Any]) -> np.ndarray:
    from app.domain.features.benchmark_frontend import log_mel_batch

    return log_mel_batch(
        X,
        sample_rate=int(requirements.get("sample_rate") or 16000),
        feature_dim=int(requirements.get("feature_dim") or 80),
        time_steps=int(requirements.get("min_sequence_length") or 100),
        # Janela de análise: o contrato vence quando declara (o AST pede
        # 25 ms = 400 amostras); sem declaração, `resolve_n_fft` a deriva do
        # salto para garantir sobreposição mínima. O `or 512` que existia aqui
        # fixava 32 ms para todo mundo — com salto de 480, sobreposição de 6%.
        n_fft=requirements.get("n_fft"),
    )


def _to_spectrogram(X: np.ndarray, requirements: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(X, dtype="float32")
    if _looks_like_raw_audio(arr):
        return _raw_audio_to_logmel(arr, requirements)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        # Vetores tabulares viram grade T x F por repetição/truncamento.
        feature_dim = int(requirements.get("feature_dim") or 80)
        time_steps = int(requirements.get("min_sequence_length") or 100)
        flat = _fit_length(arr.reshape(len(arr), -1), time_steps * feature_dim)
        arr = flat.reshape(len(arr), time_steps, feature_dim)
    elif arr.ndim == 3:
        # Piso defensivo: se o spec não declara o alvo, garante uma grade grande
        # o bastante para sobreviver às camadas de pooling (evita "Negative
        # dimension" em arquiteturas profundas com entradas sintéticas pequenas).
        time_steps = int(
            requirements.get("min_sequence_length") or max(arr.shape[1], 64)
        )
        feature_dim = int(requirements.get("feature_dim") or max(arr.shape[2], 64))
        arr = _resize_axis(arr, max(1, time_steps), axis=1)
        arr = _resize_axis(arr, max(1, feature_dim), axis=2)
    else:
        flat = arr.reshape(len(arr), -1)
        feature_dim = int(requirements.get("feature_dim") or 80)
        time_steps = int(requirements.get("min_sequence_length") or 100)
        flat = _fit_length(flat, time_steps * feature_dim)
        arr = flat.reshape(len(arr), time_steps, feature_dim)
    return _normalize_per_sample(arr)
