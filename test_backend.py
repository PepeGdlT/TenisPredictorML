"""
Test completo del backend Tennis Predictor
Comprobaciones de funcionalidad, modelo y API
"""
import unittest
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Añadir el directorio raíz al path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports del proyecto
from tennis_services import TennisPredictor, PlayerAnalyzer, TournamentAnalyzer
from app import app

class TestBackendSetup(unittest.TestCase):
    """Tests básicos de configuración del backend"""

    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todos los tests"""
        cls.outputs_dir = Path('outputs')
        cls.data_dir = Path('data')
        cls.processed_dir = cls.data_dir / 'processed'

        # Cache simulado para tests
        cls.model_cache = {}
        cls.data_cache = {}

        print("🧪 Iniciando tests del backend Tennis Predictor...")

    def test_01_directories_exist(self):
        """Verificar que existen los directorios necesarios"""
        print("📁 Test: Verificando directorios...")

        self.assertTrue(self.outputs_dir.exists(), "Directorio outputs/ debe existir")
        self.assertTrue(self.data_dir.exists(), "Directorio data/ debe existir")
        self.assertTrue(self.processed_dir.exists(), "Directorio data/processed/ debe existir")

        print("✅ Directorios verificados correctamente")

    def test_02_model_files_exist(self):
        """Verificar que existen los archivos del modelo"""
        print("🤖 Test: Verificando archivos del modelo...")

        required_files = [
            'best_xgb_model.pkl',
            'calibrated_model.pkl',
            'imputer.pkl',
            'training_states.pkl',
            'players_list.pkl',
            'model_summary.json'
        ]

        for file_name in required_files:
            file_path = self.outputs_dir / file_name
            self.assertTrue(file_path.exists(), f"Archivo {file_name} debe existir")

        print("✅ Archivos del modelo verificados")

    def test_03_data_files_exist(self):
        """Verificar que existen los archivos de datos procesados"""
        print("📊 Test: Verificando archivos de datos...")

        required_files = [
            'train_full.csv',
            'test_full.csv',
            'train_final.csv',
            'test_final.csv'
        ]

        for file_name in required_files:
            file_path = self.processed_dir / file_name
            self.assertTrue(file_path.exists(), f"Archivo {file_name} debe existir")

        print("✅ Archivos de datos verificados")

class TestModelLoading(unittest.TestCase):
    """Tests de carga del modelo y componentes"""

    @classmethod
    def setUpClass(cls):
        cls.outputs_dir = Path('outputs')

    def test_01_load_xgb_model(self):
        """Verificar que se puede cargar el modelo XGBoost"""
        print("🤖 Test: Cargando modelo XGBoost...")

        model_path = self.outputs_dir / 'best_xgb_model.pkl'

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Verificar que es un modelo válido
            self.assertTrue(hasattr(model, 'predict'), "El modelo debe tener método predict")
            self.assertTrue(hasattr(model, 'predict_proba'), "El modelo debe tener método predict_proba")

            print("✅ Modelo XGBoost cargado correctamente")

        except Exception as e:
            self.fail(f"Error cargando modelo XGBoost: {e}")

    def test_02_load_calibrated_model(self):
        """Verificar que se puede cargar el modelo calibrado"""
        print("📊 Test: Cargando modelo calibrado...")

        model_path = self.outputs_dir / 'calibrated_model.pkl'

        try:
            with open(model_path, 'rb') as f:
                calibrated_model = pickle.load(f)

            self.assertTrue(hasattr(calibrated_model, 'predict_proba'),
                          "El modelo calibrado debe tener método predict_proba")

            print("✅ Modelo calibrado cargado correctamente")

        except Exception as e:
            self.fail(f"Error cargando modelo calibrado: {e}")

    def test_03_load_training_states(self):
        """Verificar que se pueden cargar los estados de entrenamiento"""
        print("📈 Test: Cargando estados de entrenamiento...")

        states_path = self.outputs_dir / 'training_states.pkl'

        try:
            with open(states_path, 'rb') as f:
                training_states = pickle.load(f)

            # Verificar keys esperadas
            expected_keys = [
                'final_global_elos',
                'final_surface_elos',
                'final_h2h',
                'final_stats',
                'pca_state',
                'feature_columns'
            ]

            for key in expected_keys:
                self.assertIn(key, training_states, f"Debe existir la key {key}")

            print("✅ Estados de entrenamiento cargados correctamente")

        except Exception as e:
            self.fail(f"Error cargando estados de entrenamiento: {e}")

    def test_04_load_players_list(self):
        """Verificar que se puede cargar la lista de jugadores"""
        print("👥 Test: Cargando lista de jugadores...")

        players_path = self.outputs_dir / 'players_list.pkl'

        try:
            with open(players_path, 'rb') as f:
                players_list = pickle.load(f)

            self.assertIsInstance(players_list, list, "players_list debe ser una lista")
            self.assertGreater(len(players_list), 1000, "Debe haber al menos 1000 jugadores")

            # Verificar que contiene jugadores conocidos
            known_players = ['Roger Federer', 'Rafael Nadal', 'Novak Djokovic']
            for player in known_players:
                # Buscar jugadores con nombres similares (puede haber variaciones)
                found = any(player.split()[-1].lower() in p.lower() for p in players_list)
                if not found:
                    print(f"⚠️  Jugador {player} no encontrado exactamente")

            print(f"✅ Lista de jugadores cargada: {len(players_list)} jugadores")

        except Exception as e:
            self.fail(f"Error cargando lista de jugadores: {e}")

class TestDataProcessing(unittest.TestCase):
    """Tests de procesamiento de datos"""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path('data/processed')

    def test_01_load_training_data(self):
        """Verificar que se pueden cargar los datos de entrenamiento"""
        print("📊 Test: Cargando datos de entrenamiento...")

        train_path = self.data_dir / 'train_full.csv'

        try:
            df_train = pd.read_csv(train_path, low_memory=False)

            self.assertGreater(len(df_train), 100000, "Debe haber al menos 100k partidos de entrenamiento")
            self.assertIn('target', df_train.columns, "Debe existir columna target")

            # Verificar balance de clases
            target_balance = df_train['target'].value_counts()
            balance_ratio = min(target_balance) / max(target_balance)
            self.assertGreater(balance_ratio, 0.4, "Las clases deben estar relativamente balanceadas")

            print(f"✅ Datos de entrenamiento cargados: {len(df_train):,} partidos")
            print(f"   Balance de clases: {balance_ratio:.3f}")

        except Exception as e:
            self.fail(f"Error cargando datos de entrenamiento: {e}")

    def test_02_load_test_data(self):
        """Verificar que se pueden cargar los datos de test"""
        print("📊 Test: Cargando datos de test...")

        test_path = self.data_dir / 'test_full.csv'

        try:
            df_test = pd.read_csv(test_path, low_memory=False)

            self.assertGreater(len(df_test), 1000, "Debe haber al menos 1000 partidos de test")
            self.assertIn('target', df_test.columns, "Debe existir columna target")

            print(f"✅ Datos de test cargados: {len(df_test):,} partidos")

        except Exception as e:
            self.fail(f"Error cargando datos de test: {e}")

    def test_03_data_consistency(self):
        """Verificar consistencia entre datasets"""
        print("🔍 Test: Verificando consistencia de datos...")

        try:
            df_train = pd.read_csv(self.data_dir / 'train_full.csv', low_memory=False)
            df_test = pd.read_csv(self.data_dir / 'test_full.csv', low_memory=False)

            # Verificar que tienen las mismas columnas
            train_cols = set(df_train.columns)
            test_cols = set(df_test.columns)

            self.assertEqual(train_cols, test_cols, "Train y test deben tener las mismas columnas")

            # Verificar rangos temporales si existe tourney_date
            if 'tourney_date' in df_train.columns:
                train_dates = pd.to_datetime(df_train['tourney_date'], errors='coerce')
                test_dates = pd.to_datetime(df_test['tourney_date'], errors='coerce')

                max_train_date = train_dates.max()
                min_test_date = test_dates.min()

                # Test debe ser posterior a train
                if pd.notna(max_train_date) and pd.notna(min_test_date):
                    self.assertLessEqual(max_train_date, min_test_date,
                                       "Test debe contener datos posteriores a train")

            print("✅ Consistencia de datos verificada")

        except Exception as e:
            self.fail(f"Error verificando consistencia: {e}")

class TestTennisServices(unittest.TestCase):
    """Tests de los servicios de tenis"""

    @classmethod
    def setUpClass(cls):
        """Configurar servicios de test"""
        print("🔧 Configurando servicios para test...")

        # Cargar caches REALES - no simulados
        cls.model_cache = {}
        cls.data_cache = {}

        try:
            # Cargar TODOS los componentes necesarios como lo hace la app real
            outputs_dir = Path('outputs')

            # MODELO XGBOOST PRINCIPAL
            xgb_path = outputs_dir / 'best_xgb_model.pkl'
            if xgb_path.exists():
                with open(xgb_path, 'rb') as f:
                    cls.model_cache['xgb_model'] = pickle.load(f)
                print("✅ Modelo XGBoost cargado para test")
            else:
                print("⚠️ Modelo XGBoost no encontrado - test será limitado")

            # IMPUTER
            imputer_path = outputs_dir / 'imputer.pkl'
            if imputer_path.exists():
                with open(imputer_path, 'rb') as f:
                    cls.model_cache['imputer'] = pickle.load(f)
                print("✅ Imputer cargado para test")
            else:
                print("⚠️ Imputer no encontrado")

            # TRAINING STATES (esencial para TennisPredictor)
            states_path = outputs_dir / 'training_states.pkl'
            if states_path.exists():
                with open(states_path, 'rb') as f:
                    cls.model_cache['training_states'] = pickle.load(f)
                print("✅ Training states cargado para test")
            else:
                print("⚠️ Training states no encontrado")

            # PLAYERS LIST
            players_path = outputs_dir / 'players_list.pkl'
            if players_path.exists():
                with open(players_path, 'rb') as f:
                    cls.model_cache['players_list'] = pickle.load(f)
                print("✅ Players list cargado para test")
            else:
                print("⚠️ Players list no encontrado")

            # MODEL SUMMARY
            summary_path = outputs_dir / 'model_summary.json'
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    cls.model_cache['model_summary'] = json.load(f)
                print("✅ Model summary cargado para test")

            # Cargar datos históricos básicos
            data_path = Path('data/processed/train_full.csv')
            if data_path.exists():
                df_sample = pd.read_csv(data_path, nrows=1000, low_memory=False)  # Solo muestra para tests
                cls.data_cache['historical_data'] = df_sample
                print("✅ Datos históricos cargados para test")

            print("✅ Servicios configurados para test")

        except Exception as e:
            print(f"⚠️ Error configurando servicios: {e}")

    def test_01_player_analyzer_init(self):
        """Test inicialización del analizador de jugadores"""
        print("👥 Test: Inicializando PlayerAnalyzer...")

        try:
            analyzer = PlayerAnalyzer(self.data_cache)
            self.assertIsNotNone(analyzer, "PlayerAnalyzer debe inicializarse")
            print("✅ PlayerAnalyzer inicializado correctamente")

        except Exception as e:
            self.fail(f"Error inicializando PlayerAnalyzer: {e}")

    def test_02_tournament_analyzer_init(self):
        """Test inicialización del analizador de torneos"""
        print("🏆 Test: Inicializando TournamentAnalyzer...")

        try:
            analyzer = TournamentAnalyzer(self.data_cache)
            self.assertIsNotNone(analyzer, "TournamentAnalyzer debe inicializarse")
            print("✅ TournamentAnalyzer inicializado correctamente")

        except Exception as e:
            self.fail(f"Error inicializando TournamentAnalyzer: {e}")

    def test_03_tennis_predictor_init(self):
        """Test inicialización del predictor de tenis"""
        print("🎾 Test: Inicializando TennisPredictor...")

        # Verificar que tenemos los componentes mínimos necesarios
        required_components = ['xgb_model', 'imputer', 'training_states']
        missing_components = []

        for component in required_components:
            if component not in self.model_cache:
                missing_components.append(component)

        if missing_components:
            print(f"⚠️ Componentes faltantes para TennisPredictor: {missing_components}")
            print("⏭️ Saltando test - requiere modelo entrenado completo")
            self.skipTest(f"Componentes requeridos faltantes: {missing_components}")
            return

        try:
            predictor = TennisPredictor(self.model_cache, self.data_cache)
            self.assertIsNotNone(predictor, "TennisPredictor debe inicializarse")
            print("✅ TennisPredictor inicializado correctamente")

            # Test adicional: verificar que tiene el modelo cargado
            self.assertIsNotNone(predictor.model, "TennisPredictor debe tener modelo cargado")
            self.assertIsNotNone(predictor.imputer, "TennisPredictor debe tener imputer cargado")
            self.assertGreater(len(predictor.feature_columns), 0, "TennisPredictor debe tener feature columns")

            print(f"📊 Features del predictor: {len(predictor.feature_columns)}")

        except Exception as e:
            self.fail(f"Error inicializando TennisPredictor: {e}")

    def test_04_tennis_predictor_prediction(self):
        """Test de predicción básica si el predictor está disponible"""
        print("🎯 Test: Predicción básica...")

        # Solo ejecutar si tenemos todos los componentes
        required_components = ['xgb_model', 'imputer', 'training_states']
        if not all(comp in self.model_cache for comp in required_components):
            self.skipTest("Componentes requeridos no disponibles para test de predicción")
            return

        try:
            predictor = TennisPredictor(self.model_cache, self.data_cache)

            # Test de predicción simple
            prediction = predictor.predict_match(
                player1="Rafael Nadal",
                player2="Novak Djokovic",
                surface="Clay"
            )

            # Verificar estructura de la respuesta
            self.assertIn('predictions', prediction, "Respuesta debe incluir 'predictions'")
            self.assertIn('player1_win_probability', prediction['predictions'])
            self.assertIn('player2_win_probability', prediction['predictions'])
            self.assertIn('predicted_winner', prediction['predictions'])

            # Verificar que las probabilidades suman 1
            prob1 = prediction['predictions']['player1_win_probability']
            prob2 = prediction['predictions']['player2_win_probability']
            total_prob = prob1 + prob2
            self.assertAlmostEqual(total_prob, 1.0, places=2, msg="Las probabilidades deben sumar ~1.0")

            print(f"✅ Predicción exitosa: {prediction['predictions']['predicted_winner']} ({prob1:.3f} vs {prob2:.3f})")

        except Exception as e:
            self.fail(f"Error en test de predicción: {e}")

class TestAPIEndpoints(unittest.TestCase):
    """Tests de endpoints de la API"""

    @classmethod
    def setUpClass(cls):
        """Configurar cliente de test para Flask"""
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_health_endpoint(self):
        """Test endpoint de salud"""
        print("💊 Test: Endpoint /api/health...")

        response = self.client.get('/api/health')

        self.assertEqual(response.status_code, 200, "Health endpoint debe retornar 200")

        data = json.loads(response.data)
        self.assertIn('status', data, "Respuesta debe incluir status")

        print(f"✅ Health endpoint OK - Status: {data.get('status', 'unknown')}")

    def test_02_root_endpoint(self):
        """Test endpoint raíz"""
        print("🏠 Test: Endpoint /...")

        response = self.client.get('/')

        self.assertEqual(response.status_code, 200, "Root endpoint debe retornar 200")

        data = json.loads(response.data)
        self.assertIn('message', data, "Respuesta debe incluir message")
        self.assertEqual(data['message'], 'Tennis Predictor API')

        print("✅ Root endpoint OK")

    def test_03_players_endpoint(self):
        """Test endpoint de jugadores"""
        print("👥 Test: Endpoint /api/players...")

        response = self.client.get('/api/players?limit=10')

        # Puede retornar 200 o 503 si no hay datos
        self.assertIn(response.status_code, [200, 503],
                     "Players endpoint debe retornar 200 o 503")

        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('players', data, "Respuesta debe incluir players")
            print(f"✅ Players endpoint OK - {len(data.get('players', []))} jugadores")
        else:
            print("⚠️  Players endpoint retornó 503 (servicios no disponibles)")

    def test_04_surfaces_endpoint(self):
        """Test endpoint de superficies"""
        print("🎾 Test: Endpoint /api/surfaces...")

        response = self.client.get('/api/surfaces')

        self.assertEqual(response.status_code, 200, "Surfaces endpoint debe retornar 200")

        data = json.loads(response.data)
        self.assertIn('surfaces', data, "Respuesta debe incluir surfaces")

        expected_surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
        for surface in expected_surfaces:
            self.assertIn(surface, data['surfaces'], f"Debe incluir superficie {surface}")

        print("✅ Surfaces endpoint OK")

class TestPredictionLogic(unittest.TestCase):
    """Tests de lógica de predicción básica"""

    def test_01_prediction_request_format(self):
        """Test formato de request de predicción"""
        print("🎯 Test: Formato de request de predicción...")

        app.config['TESTING'] = True
        client = app.test_client()

        # Test con datos válidos
        prediction_data = {
            'player1': 'Roger Federer',
            'player2': 'Rafael Nadal',
            'surface': 'Clay'
        }

        response = client.post('/api/predict',
                             data=json.dumps(prediction_data),
                             content_type='application/json')

        # Puede retornar 200 o 503 dependiendo de si el modelo está cargado
        self.assertIn(response.status_code, [200, 400, 503],
                     "Predict endpoint debe manejar requests apropiadamente")

        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('predictions', data, "Respuesta debe incluir predictions")
            print("✅ Prediction request format OK")
        else:
            print(f"⚠️  Prediction endpoint retornó {response.status_code}")

def run_comprehensive_test():
    """Ejecutar todos los tests de manera comprehensiva"""

    print("=" * 80)
    print("🧪 INICIANDO TESTS COMPREHENSIVOS DEL BACKEND")
    print("=" * 80)

    # Crear test suite
    test_classes = [
        TestBackendSetup,
        TestModelLoading,
        TestDataProcessing,
        TestTennisServices,
        TestAPIEndpoints,
        TestPredictionLogic
    ]

    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("📊 RESUMEN DE TESTS")
    print("=" * 80)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    print(f"Éxito: {result.wasSuccessful()}")

    if result.failures:
        print("\n❌ FALLOS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\n💥 ERRORES:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    if result.wasSuccessful():
        print("\n🎉 ¡TODOS LOS TESTS PASARON EXITOSAMENTE!")
    else:
        print(f"\n⚠️  {len(result.failures + result.errors)} tests fallaron")

    print("=" * 80)

    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
