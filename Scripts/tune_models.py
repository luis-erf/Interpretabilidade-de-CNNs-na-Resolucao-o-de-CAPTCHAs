import argparse
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import pandas as pd

from Dataset.Datamodule import CaptchaDataModule
from Models.ConvNext import ConvNeXtCaptcha
from Models.Vgg import VGGCaptcha

# =====================================================================
# 1. DICIONÁRIO DE ESPAÇO DE BUSCA (SEARCH SPACE)
# =====================================================================
# Você pode adicionar, remover ou alterar os limites dos hiperparâmetros aqui.
SEARCH_SPACE = {
    "VGGCaptcha": {
        "learning_rate": {"type": "log_float", "low": 1e-4, "high": 1e-2},
        "dropout_rate": {"type": "float", "low": 0.2, "high": 0.6},
        # Representamos as camadas densas como strings para o Optuna escolher,
        # e depois convertemos para listas dentro da função objective.
        "fc_sizes_str": {"type": "categorical", "choices": ["512", "1024", "512_512"]}
    },
    "ConvNeXtCaptcha": {
        "learning_rate": {"type": "log_float", "low": 1e-4, "high": 1e-2},
        "weight_decay": {"type": "log_float", "low": 1e-5, "high": 1e-2},
        "expansion_ratio": {"type": "categorical", "choices": [2, 4, 6]}
    }
}
# =====================================================================

def parse_fc_sizes(fc_str):
    """Função auxiliar para converter a escolha da camada densa da VGG de string para lista"""
    return [int(x) for x in fc_str.split('_')]

def objective(trial, model_name, datamodule, test_mode=False):
    """
    Função objetivo que o Optuna tenta otimizar (minimizar a perda de validação).
    """
    # 2. Extrair hiperparâmetros sugeridos pelo Optuna para esta iteração (trial)
    params = {}
    for param_name, config in SEARCH_SPACE[model_name].items():
        p_type = config["type"]
        if p_type == "log_float":
            params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], log=True)
        elif p_type == "float":
            params[param_name] = trial.suggest_float(param_name, config["low"], config["high"])
        elif p_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, config["choices"])

    # 3. Instanciar o modelo correspondente com os hiperparâmetros escolhidos
    if model_name == "VGGCaptcha":
        # Converte a string sugerida de volta para a lista (ex: "512_512" -> [512, 512])
        params["fc_sizes"] = parse_fc_sizes(params.pop("fc_sizes_str"))
        model = VGGCaptcha(**params)
        
    elif model_name == "ConvNeXtCaptcha":
        model = ConvNeXtCaptcha(**params)
    else:
        raise ValueError(f"Modelo {model_name} não reconhecido.")

    # 4. Configurar Early Stopping
    # Interrompe o treinamento se a 'val_loss' não melhorar por 3 épocas consecutivas
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=False
    )

    # 5. Configurar o Trainer do PyTorch Lightning
    trainer_kwargs = {
        "max_epochs": 15 if not test_mode else 1, # Apenas 1 época se for teste de sanidade
        "callbacks": [early_stop],
        "enable_progress_bar": False, # Desliga a barra para não poluir o terminal na tunagem
        "logger": False, # Evita criar centenas de pastas de logs no disco
        "enable_model_summary": False
    }

    # Se estiver em TEST_MODE, limita os dados para o código rodar em segundos
    if test_mode:
        trainer_kwargs["limit_train_batches"] = 2
        trainer_kwargs["limit_val_batches"] = 2

    # Inicializa o treinador
    trainer = pl.Trainer(**trainer_kwargs)

    # 6. Iniciar o Treinamento
    trainer.fit(model, datamodule=datamodule)

    # 7. Recuperar a melhor métrica de validação obtida pelo modelo
    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is None:
        return float('inf') # Se falhar, retorna infinito para o Optuna descartar a tentativa
        
    return val_loss.item()


def run_tuning(models_to_tune, datamodule, n_trials=15, test_mode=False):
    """
    Executa a tunagem para uma lista de modelos e retorna um dicionário com os melhores resultados.
    """
    melhores_resultados = {}

    for model_name in models_to_tune:
        print(f"\n{'='*50}")
        print(f" Iniciando Tunagem para: {model_name} ")
        print(f"{'='*50}")

        # Cria um 'study' (estudo) no Optuna com o objetivo de minimizar o erro
        study = optuna.create_study(direction="minimize", study_name=f"Tuning_{model_name}")
        
        # Ajusta a quantidade de iterações do Optuna se for um teste rápido
        trials_to_run = 2 if test_mode else n_trials

        # Executa a otimização
        study.optimize(lambda trial: objective(trial, model_name, datamodule, test_mode), 
                       n_trials=trials_to_run)

        # Salva os melhores hiperparâmetros deste modelo
        melhores_resultados[model_name] = study.best_params
        
        print(f"\n--> Tunagem de {model_name} concluída!")
        print(f"--> Menor Val Loss: {study.best_value:.4f}")
        print(f"--> Melhores Hiperparâmetros: {study.best_params}")

    return melhores_resultados


if __name__ == "__main__":
    # =====================================================================
    # CONTROLE PRINCIPAL VIA TERMINAL
    # =====================================================================
    parser = argparse.ArgumentParser(description="Script de Tunagem de Modelos para CAPTCHA")
    parser.add_argument('--test', action='store_true', help="Ativa o modo de teste rápido (Sanity Check)")
    args = parser.parse_args()

    # A variável agora recebe o valor passado no terminal (True se --test existir, False caso contrário)
    TEST_MODE = args.test
    
    # Define quais modelos do SEARCH_SPACE queremos tunar agora
    MODELOS_PARA_TUNAR = ["VGGCaptcha", "ConvNeXtCaptcha"]
    
    print(f"Modo de Teste (Sanity Check): {'ATIVADO' if TEST_MODE else 'DESATIVADO'}")
    
    # Instancia o DataModule (Ajuste o caminho da pasta conforme seu gerador)
    datamodule = CaptchaDataModule(
        diretorio_dados="dataset_captcha", 
        batch_size=32, 
        num_workers=2
    )
    
    # Chama a função principal de tunagem
    resultados_finais = run_tuning(
        models_to_tune=MODELOS_PARA_TUNAR, 
        datamodule=datamodule, 
        n_trials=20, # Quantidade de tentativas reais no Optuna por modelo (ignorado no modo teste)
        test_mode=TEST_MODE
    )
    
    # Exibe o dicionário final exigido
    print("\n\n" + "#"*50)
    print(" RESUMO FINAL DA TUNAGEM ")
    print("#"*50)
    for modelo, hiperparametros in resultados_finais.items():
        print(f"\nModelo: {modelo}")
        for param, valor in hiperparametros.items():
            print(f"  - {param}: {valor}")
    df = pd.DataFrame(resultados_finais)
    df.to_csv('tune_results.csv')