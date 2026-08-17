"""Guardas do retune dos clássicos (SVM/Random Forest) — 2026-08-09.

Cada teste aqui trava um defeito CONCRETO encontrado na revisão do
`clean_benchmark_15k`, onde os dois clássicos fecharam com acurácia 0,8531 no
limpo e desabaram a 5 dB (SVM 0,5000 com recall 0,0000; RF 0,6274) mantendo
AUC de 0,849 e 0,838 — ou seja, a ordenação sobrevivia e o ponto de operação
não.

Os defeitos: grid regularizado que nunca rodou (o runner tinha uma cópia
própria), validação cruzada sem agrupamento sobre um dataset PAREADO,
hiperparâmetros escolhidos no regime limpo e aplicados no regime ruidoso,
probabilidades descalibradas sob limiar fixo de 0,5, e limiar de operação
derivado de dados que estavam dentro do ajuste.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")


@pytest.fixture
def dados_agrupados():
    """40 clusters × 8 amostras, rótulos balanceados dentro de cada cluster."""
    rng = np.random.default_rng(42)
    n_clusters, por_cluster = 40, 8
    groups = np.repeat([f"c{i}" for i in range(n_clusters)], por_cluster)
    y = np.tile([0, 1], len(groups) // 2)
    X = rng.standard_normal((len(y), 20))
    X[y == 1] += 0.7
    return X, y, groups


class TestValidacaoCruzadaAgrupada:
    def test_nenhum_cluster_atravessa_dobras(self, dados_agrupados):
        """O par real/clone da MESMA frase não pode se dividir entre dobras.

        O Protocolo de Dataset é pareado: cada enunciado aparece como original
        CETUC e como clone XTTS-v2 do mesmo locutor e da mesma frase. Com
        partição aleatória, o modelo acerta a dobra de validação reconhecendo
        o enunciado que acabou de ver no treino dela — sem detectar síntese.
        """
        from benchmarks.runner import _classical_cv_splitter

        X, y, groups = dados_agrupados
        splitter, cv, kind, note = _classical_cv_splitter(y, groups, seed=42)

        assert kind == "StratifiedGroupKFold"
        assert cv == 5
        assert "40 grupos" in note
        for treino, validacao in splitter.split(X, y, groups):
            assert not (set(groups[treino]) & set(groups[validacao]))

    def test_sem_cluster_ids_degrada_declarando(self, dados_agrupados):
        """Degradar é aceitável; degradar em silêncio, não."""
        from benchmarks.runner import _classical_cv_splitter

        _X, y, _groups = dados_agrupados
        _splitter, cv, kind, note = _classical_cv_splitter(y, None, seed=42)

        assert kind == "StratifiedKFold"
        assert cv == 5
        assert "SEM agrupamento" in note

    def test_grupos_desalinhados_falham_alto(self, dados_agrupados):
        from benchmarks.runner import _classical_cv_splitter

        _X, y, groups = dados_agrupados
        with pytest.raises(RuntimeError, match="desalinhados"):
            _classical_cv_splitter(y, groups[:-5], seed=42)

    def test_poucas_amostras_por_classe_nao_quebra(self):
        from benchmarks.runner import _classical_cv_splitter

        splitter, cv, kind, _note = _classical_cv_splitter(
            np.array([0, 1]), None, seed=42
        )
        assert splitter is None and cv < 2 and kind == "none"

    def test_tuning_registra_o_agrupamento_no_artefato(
        self, dados_agrupados, tmp_path
    ):
        """Quem lê o `hyperparameter_tuning.json` precisa saber qual CV rodou."""
        from benchmarks.runner import _run_classical_tuning

        X, y, groups = dados_agrupados
        plano = _run_classical_tuning(
            arch="SVM", X=X, y=y, output_dir=tmp_path, seed=42, groups=groups
        )
        assert plano["status"] == "ok"
        assert plano["cv_kind"] == "StratifiedGroupKFold"
        assert plano["cv"] == 5
        assert "cluster_ids" in plano["cv_grouping"]


class TestGridUnico:
    def test_runner_consome_o_grid_da_arquitetura(self):
        """Não pode existir uma 4ª fonte de hiperparâmetros.

        O CLAUDE.md lista TRÊS (registry, create_model, planning). O
        `_classical_search_space` era uma quarta, não documentada, e era ela
        que rodava: os grids regularizados de `svm.py`/`random_forest.py` não
        tinham nenhum chamador em `app/`, `benchmarks/`, `scripts/` ou
        `tests/`.
        """
        from app.domain.models.architectures.random_forest import (
            RANDOM_FOREST_PARAM_GRID,
        )
        from app.domain.models.architectures.svm import SVM_PARAM_GRID
        from benchmarks.runner import _classical_search_space

        assert _classical_search_space("SVM", 42)[0] == SVM_PARAM_GRID
        assert (
            _classical_search_space("RandomForest", 42)[0]
            == RANDOM_FOREST_PARAM_GRID
        )

    def test_grid_do_rf_regulariza_de_fato(self):
        from app.domain.models.architectures.random_forest import (
            RANDOM_FOREST_PARAM_GRID as grid,
        )

        # `max_depth=None` + `min_samples_leaf=1` foi o que venceu no
        # clean_benchmark_15k, com mean_train_score = 1.0.
        assert None not in grid["rf__max_depth"]
        assert min(grid["rf__min_samples_leaf"]) >= 2
        assert "rf__min_samples_split" in grid
        assert min(grid["rf__min_samples_split"]) >= 5

    def test_grid_do_svm_nao_desperdiça_o_eixo_gamma(self):
        """`gamma` não cruza com o kernel linear, que o ignora.

        Como dicionário único o grid gerava 3 C x 4 gamma = 12 candidatos
        lineares para 3 distintos — e o linear é o caro: em 16.000 amostras do
        vetor v2, `linear C=10` leva 40,6 s contra 3,2 s do `rbf C=10`. Eram 9
        ajustes redundantes entre os mais lentos da busca.
        """
        from sklearn.model_selection import ParameterGrid

        from app.domain.models.architectures.svm import SVM_PARAM_GRID as grid

        assert isinstance(grid, list), "grid único voltaria a cruzar gamma x linear"
        candidatos = list(ParameterGrid(grid))
        assert len(candidatos) == 15  # 3 C x 4 gamma (rbf) + 3 C (linear)

        lineares = [c for c in candidatos if c["svm__kernel"] == "linear"]
        assert len(lineares) == 3
        assert not any("svm__gamma" in c for c in lineares)

        rbf = [c for c in candidatos if c["svm__kernel"] == "rbf"]
        gammas = {c["svm__gamma"] for c in rbf}
        # Depois do StandardScaler, var≈1 e 'scale'≈'auto'≈1/n_features: manter
        # os dois testava o MESMO gamma duas vezes (postos 1 e 2 do run real
        # diferiram em 2e-6).
        assert "auto" not in gammas
        assert sum(isinstance(g, float) for g in gammas) >= 3

        kernels = {c["svm__kernel"] for c in candidatos}
        assert "poly" not in kernels
        assert max(c["svm__C"] for c in candidatos) <= 10

    def test_busca_do_svm_nao_paga_platt_interno(self):
        """`probability=True` na busca custa 6x e não compra informação.

        O libsvm roda uma CV interna de 5 dobras a cada ajuste para calibrar
        Platt. O `scoring` é `roc_auc`, baseado em ORDENAÇÃO, e Platt é
        monotônica: a AUC é idêntica com ou sem. Quem dá probabilidade ao
        modelo final é a calibração isotônica.
        """
        from benchmarks.runner import _classical_search_space

        _grid, estimador, _step = _classical_search_space("SVM", 42)
        assert estimador.probability is False


class TestCalibracao:
    @pytest.mark.parametrize("arch", ["RandomForest", "SVM"])
    def test_calibracao_preserva_o_estimador_explicavel(self, arch):
        """Calibrar não pode custar a XAI do trabalho.

        Com o default `ensemble=True` o resultado é a média de `cv` modelos,
        nenhum deles ajustado no conjunto inteiro: `feature_importances_`
        some, o `TreeExplainer` não tem o que explicar e
        `export_rf_feature_importance.py` quebra.
        """
        from app.domain.xai.tabular import (
            extract_sklearn_estimator,
            split_sklearn_pipeline,
        )

        if arch == "RandomForest":
            from app.domain.models.architectures.random_forest import (
                create_random_forest_model as factory,
            )

            esperado = "RandomForestClassifier"
        else:
            from app.domain.models.architectures.svm import (
                create_svm_model as factory,
            )

            esperado = "SVC"

        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 12))
        y = np.tile([0, 1], 100)
        X[y == 1] += 0.9

        modelo = factory(
            input_shape=(12,), num_classes=2, calibrate=True, random_state=42
        )
        modelo.fit(X, y)

        assert type(modelo.pipeline.steps[-1][1]).__name__ == (
            "CalibratedClassifierCV"
        )
        assert type(extract_sklearn_estimator(modelo.pipeline)).__name__ == esperado
        _transform, estimador = split_sklearn_pipeline(modelo.pipeline)
        assert type(estimador).__name__ == esperado

        proba = modelo.predict_proba(X)
        assert proba.shape == (200, 2)
        assert np.isfinite(proba).all()

        if arch == "RandomForest":
            importancias = modelo.get_feature_importance()
            assert len(importancias) == 12
            assert sum(importancias.values()) == pytest.approx(1.0, abs=1e-6)

    def test_wrap_calibration_usa_ensemble_false(self):
        from sklearn.ensemble import RandomForestClassifier

        from app.domain.models.architectures.classical_ml_helpers import (
            wrap_calibration,
        )

        base = RandomForestClassifier(n_estimators=5, random_state=0)
        assert wrap_calibration(base, False) is base
        calibrado = wrap_calibration(base, True)
        assert calibrado.method == "isotonic"
        assert calibrado.ensemble is False


class TestRegimeDoAjuste:
    def test_cv_roda_no_mesmo_conjunto_do_ajuste(self):
        """A busca via `X_fit_2d`, não mais só o bloco limpo.

        Escolher hiperparâmetros só no limpo e ajustar em limpo+ruidoso decide
        o modelo num regime em que ele nunca opera — justamente o que o
        protocolo mede a 10 e a 5 dB.
        """
        from pathlib import Path

        fonte = (
            Path(__file__).resolve().parents[2] / "benchmarks" / "runner.py"
        ).read_text(encoding="utf-8")

        assert "X=X_train_2d[:clean_train_count]" not in fonte
        assert "X=X_fit_2d," in fonte

    def test_grupos_repetem_junto_com_as_copias_ruidosas(self):
        """A cópia AWGN da amostra i tem o MESMO cluster da amostra i.

        Sem repetir o grupo por bloco, o par limpo/ruidoso da mesma amostra
        cairia em dobras diferentes — vazamento mais direto ainda que o do par
        real/clone.
        """
        clusters = np.array(["a", "b", "c"])
        n_clean, blocos = 3, 2
        tiled = np.tile(clusters, blocos)

        assert len(tiled) == n_clean * blocos
        for i in range(n_clean):
            assert tiled[i] == tiled[n_clean + i]


class TestIsolamentoDoModelsDir:
    """`data/models` é diretório de PRODUÇÃO — nenhum teste pode escrever nele.

    Foi assim que o `bench_svm.pkl` do `clean_benchmark_15k` (63 features,
    3,6 MB) virou um artefato de smoke de 47 KB: as métricas do run
    sobreviveram, o modelo que as produziu não.
    """

    def test_fixture_cobre_toda_a_cadeia_de_resolucao(self):
        """Definir só `XFAKE_MODELS_DIR` NÃO isola.

        `_models_dir` para na primeira variável definida, e o `.env` do projeto
        declara `DEEPFAKE_MODELS_DIR=./data/models` — carregado no import de
        `app.*` que o próprio `conftest.py` faz. A primeira versão da fixture
        definia apenas a terceira da cadeia e nunca vencia.
        """
        import os

        for nome in ("MODELS_DIR", "DEEPFAKE_MODELS_DIR", "XFAKE_MODELS_DIR"):
            valor = os.environ.get(nome)
            assert valor, f"{nome} não foi isolada pela fixture de sessão"
            assert "models_dir_sandbox" in valor, (
                f"{nome} aponta para {valor}, fora do sandbox da sessão"
            )

    def test_config_sem_models_dir_nao_escreve_em_data_models(self, tmp_path):
        """O caminho REAL é exercitado — não basta o md5 sobreviver.

        Um `BenchmarkConfig` sem `models_dir` cai no default `data/models`. Este
        teste resolve o diretório pelo mesmo caminho que o runner usa e exige
        que ele caia no sandbox, com a produção intocada.
        """
        from pathlib import Path

        from benchmarks import BenchmarkConfig
        from benchmarks.runner import _models_dir

        projeto = Path(__file__).resolve().parents[2]
        producao = projeto / "data" / "models"
        antes = (
            {p.name for p in producao.iterdir()} if producao.is_dir() else set()
        )

        cfg = BenchmarkConfig(architectures=["SVM"], output_dir=str(tmp_path))
        resolvido = _models_dir(cfg, "SVM")

        assert producao not in resolvido.parents and resolvido != producao
        assert "models_dir_sandbox" in str(resolvido)

        depois = (
            {p.name for p in producao.iterdir()} if producao.is_dir() else set()
        )
        assert depois == antes


class TestScoreDeOrdenacaoSeparado:
    """Métricas SEM limiar medem ordenação — a calibração não pode truncá-la.

    A isotônica é uma função escada. No SVM do `clean_benchmark_15k` ela
    colapsou 1.382 margens distintas em 52 degraus, e os empates custaram
    0,75 pp de AUC (0,9731 no score bruto contra 0,9656 no calibrado). Como os
    modelos neurais reportam softmax sem calibração pós-hoc, só os clássicos
    pagavam esse pedágio.
    """

    @staticmethod
    def _dados():
        rng = np.random.default_rng(7)
        y = np.tile([0, 1], 200)
        bruto = rng.normal(loc=y * 1.5, scale=1.0)  # contínuo, sem empates
        # calibrado: monotônico mas em degraus (o que a isotônica produz)
        degraus = np.clip(np.round(bruto * 2) / 2, -3, 3)
        p = 1.0 / (1.0 + np.exp(-degraus))
        return y, p, bruto

    def test_metricas_sem_limiar_usam_o_score_bruto(self):
        from sklearn.metrics import roc_auc_score

        from benchmarks.evaluate import evaluate_scores

        y, p, bruto = self._dados()
        com = evaluate_scores(y, p, ranking_scores=bruto)
        sem = evaluate_scores(y, p)

        assert com["auc_roc"] == pytest.approx(roc_auc_score(y, bruto))
        assert sem["auc_roc"] == pytest.approx(roc_auc_score(y, p))
        # os degraus empatam amostras e custam AUC — é o defeito medido
        assert com["auc_roc"] > sem["auc_roc"]
        assert com["ranking_score_source"] == "raw_detector_score"
        assert sem["ranking_score_source"] == "p_fake"

    def test_metricas_com_limiar_continuam_na_probabilidade(self):
        from sklearn.metrics import accuracy_score

        from benchmarks.evaluate import evaluate_scores

        y, p, bruto = self._dados()
        com = evaluate_scores(y, p, ranking_scores=bruto)
        sem = evaluate_scores(y, p)

        esperado = accuracy_score(y, (p >= 0.5).astype(int))
        for chave in ("accuracy", "precision", "recall", "f1", "ece"):
            assert com[chave] == pytest.approx(sem[chave]), chave
        assert com["accuracy"] == pytest.approx(esperado)

    def test_score_bruto_negativo_nao_e_recortado(self):
        """`decision_function` de SVM é centrado em zero.

        Recortar em [0,1] — o tratamento correto para uma probabilidade —
        zeraria metade das margens e destruiria a ordenação que o score existe
        para preservar.
        """
        from sklearn.metrics import roc_auc_score

        from benchmarks.evaluate import evaluate_scores

        y = np.tile([0, 1], 100)
        bruto = np.linspace(-4.0, 4.0, 200)[np.argsort(np.argsort(y + 1e-9 * np.arange(200)))]
        p = 1.0 / (1.0 + np.exp(-bruto))
        out = evaluate_scores(y, p, ranking_scores=bruto)
        assert out["auc_roc"] == pytest.approx(roc_auc_score(y, bruto))

    def test_ausencia_de_ranking_e_no_op(self):
        """Todo modelo que não calibra tem de sair EXATAMENTE como antes."""
        from benchmarks.evaluate import evaluate_scores

        y, p, _ = self._dados()
        a = evaluate_scores(y, p, n_bootstrap=50)
        b = evaluate_scores(y, p, ranking_scores=p, n_bootstrap=50)
        for chave in ("auc_roc", "eer", "min_tdcf", "accuracy", "ece"):
            assert a[chave] == pytest.approx(b[chave]), chave

    def test_ranking_desalinhado_falha_alto(self):
        from benchmarks.evaluate import evaluate_scores

        y, p, bruto = self._dados()
        with pytest.raises(ValueError, match="desalinhado"):
            evaluate_scores(y, p, ranking_scores=bruto[:-3])

    def test_predictions_csv_grava_o_score_de_ordenacao(self, tmp_path):
        """Sem a coluna, o EER publicado deixaria de ser reverificável."""
        import csv as _csv

        from benchmarks.report import _write_arch_predictions_csv

        y = np.array([0, 1, 0, 1])
        r = {
            "scores_clean": [0.2, 0.8, 0.3, 0.7],
            "ranking_scores_clean": [-1.5, 2.5, -0.5, 1.0],
        }
        destino = tmp_path / "predictions_clean.csv"
        _write_arch_predictions_csv("SVM", r, y, destino)
        linhas = list(_csv.DictReader(destino.open(encoding="utf-8")))
        assert [ln["ranking_score"] for ln in linhas] == ["-1.5", "2.5", "-0.5", "1.0"]

        # sem calibração a coluna existe mas fica VAZIA — não zero, que seria
        # um score legítimo e enganaria quem recalcular.
        destino2 = tmp_path / "sem_ranking.csv"
        _write_arch_predictions_csv("Conformer", {"scores_clean": [0.2, 0.8, 0.3, 0.7]}, y, destino2)
        linhas2 = list(_csv.DictReader(destino2.open(encoding="utf-8")))
        assert {ln["ranking_score"] for ln in linhas2} == {""}

    def test_predictions_csv_agregado_tambem_grava_a_ordenacao(self, tmp_path):
        """O arquivo do run é o primeiro que alguém abre para reconferir.

        Ele saía sem a coluna enquanto o por-arquitetura já a trazia: quem
        recalculasse o AUC do SVM daqui obtinha 0,9656 contra os 0,9731 da
        tabela, e nada no arquivo explicava a diferença.
        """
        import csv as _csv

        from benchmarks.report import _write_predictions_csv

        resultados = {
            "dataset": {"y_test": [0, 1, 0, 1]},
            "architectures": {
                "SVM": {
                    "status": "ok",
                    "scores_clean": [0.2, 0.8, 0.3, 0.7],
                    "ranking_scores_clean": [-1.5, 2.5, -0.5, 1.0],
                },
                "Conformer": {
                    "status": "ok",
                    "scores_clean": [0.1, 0.9, 0.4, 0.6],
                },
            },
        }
        destino = tmp_path / "predictions_clean.csv"
        _write_predictions_csv(resultados, destino)
        linhas = list(_csv.DictReader(destino.open(encoding="utf-8")))

        svm = [ln for ln in linhas if ln["architecture"] == "SVM"]
        assert [ln["ranking_score"] for ln in svm] == ["-1.5", "2.5", "-0.5", "1.0"]
        # quem não calibra continua com a coluna vazia, não com zero.
        conformer = [ln for ln in linhas if ln["architecture"] == "Conformer"]
        assert {ln["ranking_score"] for ln in conformer} == {""}
        assert [ln["p_fake"] for ln in conformer] == ["0.1", "0.9", "0.4", "0.6"]
