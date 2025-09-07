"""
Script para hacer llamadas reales a la API del backend de tenis
Prueba todos los endpoints disponibles con datos reales
"""
import requests
import json
import time
from datetime import datetime

# Configuración de la API
BASE_URL = "http://127.0.0.1:5000"
TIMEOUT = 30

def test_api_endpoint(endpoint, method='GET', data=None, description=""):
    """
    Función helper para probar endpoints
    """
    url = f"{BASE_URL}{endpoint}"

    print(f"\n{'='*60}")
    print(f"🔍 PROBANDO: {method} {endpoint}")
    if description:
        print(f"📝 {description}")
    print(f"{'='*60}")

    try:
        start_time = time.time()

        if method == 'GET':
            response = requests.get(url, timeout=TIMEOUT)
        elif method == 'POST':
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)

        end_time = time.time()
        response_time = (end_time - start_time) * 1000

        print(f"⏱️  Tiempo de respuesta: {response_time:.2f}ms")
        print(f"📊 Status Code: {response.status_code}")

        if response.status_code == 200:
            print("✅ ÉXITO")
            try:
                response_data = response.json()

                # Mostrar información relevante según el endpoint
                if endpoint == "/":
                    print(f"🎾 Aplicación: {response_data.get('message', 'N/A')}")
                    print(f"📋 Endpoints disponibles: {len(response_data.get('endpoints', {}))}")

                elif endpoint == "/api/health":
                    print(f"💚 Estado: {response_data.get('status', 'N/A')}")
                    print(f"🔧 Modelos cargados: {response_data.get('models_loaded', 'N/A')}")
                    print(f"📊 Datos disponibles: {response_data.get('data_available', 'N/A')}")

                elif "/api/players" in endpoint:
                    players = response_data.get('players', [])
                    total = response_data.get('total', 0)
                    showing = response_data.get('showing', 0)
                    print(f"👥 Total jugadores: {total}")
                    print(f"🔢 Mostrando: {showing}")
                    if players:
                        print(f"📋 Algunos jugadores: {', '.join(players[:5])}...")

                elif endpoint == "/api/surfaces":
                    surfaces = response_data.get('surfaces', [])
                    print(f"🏟️  Superficies disponibles: {', '.join(surfaces)}")

                elif endpoint == "/api/predict":
                    predictions = response_data.get('predictions', {})
                    if predictions:
                        print(f"🏆 Ganador predicho: {predictions.get('predicted_winner', 'N/A')}")
                        print(f"🎯 Prob. Player 1: {predictions.get('player1_win_probability', 0):.3f}")
                        print(f"🎯 Prob. Player 2: {predictions.get('player2_win_probability', 0):.3f}")
                        print(f"📈 Confianza: {predictions.get('confidence', 0):.3f}")

                elif "/api/player/" in endpoint:
                    basic_stats = response_data.get('basic_stats', {})
                    if basic_stats:
                        print(f"🏆 Total partidos: {basic_stats.get('total_matches', 0)}")
                        print(f"✅ Victorias: {basic_stats.get('wins', 0)}")
                        print(f"📊 Win rate: {basic_stats.get('win_rate', 0):.3f}")

                elif endpoint == "/api/model/info":
                    performance = response_data.get('performance', {})
                    if performance:
                        print(f"🎯 Accuracy: {performance.get('accuracy', 0):.3f}")
                        print(f"📈 AUC: {performance.get('auc', 0):.3f}")

                elif endpoint == "/api/model/features":
                    features = response_data.get('features', [])
                    total_features = response_data.get('total_features', 0)
                    print(f"🔢 Total features: {total_features}")
                    if features:
                        print(f"🏅 Top 3 features:")
                        for i, feature in enumerate(features[:3], 1):
                            print(f"   {i}. {feature.get('Feature', 'N/A')}: {feature.get('Importance', 0):.3f}")

                elif "/api/tournaments" in endpoint:
                    tournaments = response_data.get('tournaments', [])
                    total = response_data.get('total', 0)
                    print(f"🏆 Total torneos: {total}")
                    if tournaments:
                        print(f"🎾 Top 3 torneos:")
                        for i, tournament in enumerate(tournaments[:3], 1):
                            print(f"   {i}. {tournament.get('name', 'N/A')}: {tournament.get('total_matches', 0)} partidos")

                elif endpoint == "/api/rankings/top":
                    top_players = response_data.get('top_players', [])
                    print(f"🏅 Top 5 jugadores históricos:")
                    for i, player in enumerate(top_players[:5], 1):
                        print(f"   {i}. {player.get('player', 'N/A')}: {player.get('wins', 0)} victorias")

                elif endpoint == "/api/stats/overview":
                    print(f"📊 Total partidos: {response_data.get('total_matches', 0):,}")
                    print(f"👥 Jugadores únicos: {response_data.get('unique_players', 0):,}")
                    print(f"🏆 Torneos únicos: {response_data.get('unique_tournaments', 0):,}")
                    date_range = response_data.get('date_range', {})
                    if date_range:
                        print(f"📅 Período: {date_range.get('earliest', 'N/A')} - {date_range.get('latest', 'N/A')}")

                # Mostrar JSON completo solo si es pequeño
                if len(str(response_data)) < 500:
                    print(f"📄 Respuesta completa:")
                    print(json.dumps(response_data, indent=2, ensure_ascii=False))
                else:
                    print(f"📄 Respuesta muy grande ({len(str(response_data))} chars) - solo mostrando resumen")

            except json.JSONDecodeError:
                print("⚠️  Respuesta no es JSON válido")
                print(f"📝 Contenido: {response.text[:200]}...")

        else:
            print(f"❌ ERROR - Status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"🔍 Error: {error_data.get('error', 'Error desconocido')}")
            except:
                print(f"📝 Contenido: {response.text[:200]}...")

    except requests.exceptions.Timeout:
        print(f"⏰ TIMEOUT - El endpoint tardó más de {TIMEOUT}s en responder")
    except requests.exceptions.ConnectionError:
        print("🔌 ERROR DE CONEXIÓN - ¿Está ejecutándose el servidor?")
    except Exception as e:
        print(f"💥 ERROR INESPERADO: {str(e)}")

def main():
    """
    Función principal que ejecuta todos los tests de la API
    """
    print("🎾 TENNIS PREDICTOR API - TESTS COMPLETOS")
    print(f"🌐 URL Base: {BASE_URL}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Test básico de conectividad
    test_api_endpoint("/", description="Endpoint principal de la aplicación")

    # 2. Health check
    test_api_endpoint("/api/health", description="Estado de salud del sistema")

    # 3. Lista de jugadores
    test_api_endpoint("/api/players?limit=10", description="Lista de jugadores (limitado a 10)")

    # 4. Búsqueda de jugadores específicos
    test_api_endpoint("/api/players?search=federer&limit=5", description="Buscar jugadores con 'federer'")

    # 5. Información de un jugador específico
    test_api_endpoint("/api/player/Rafael Nadal", description="Información detallada de Rafael Nadal")

    # 6. Superficies disponibles
    test_api_endpoint("/api/surfaces", description="Tipos de superficie disponibles")

    # 7. Predicción de partido
    prediction_data = {
        "player1": "Carlos Alcaraz",
        "player2": "Jannick Sinner",
        "surface": "Clay",
        "tournament_level": "ATP1000"
    }
    test_api_endpoint("/api/predict", method="POST", data=prediction_data,
                     description="Predicción: Alcaraz vs Sinner en arcilla")

    # 8. Otra predicción
    prediction_data2 = {
        "player1": "Roger Federer",
        "player2": "Andy Murray",
        "surface": "Grass",
        "tournament_level": "ATP500"
    }
    test_api_endpoint("/api/predict", method="POST", data=prediction_data2,
                     description="Predicción: Federer vs Murray en césped")

    # 9. Información del modelo
    test_api_endpoint("/api/model/info", description="Información del modelo ML")

    # 10. Features del modelo
    test_api_endpoint("/api/model/features", description="Feature importance del modelo")

    # 11. Lista de torneos
    test_api_endpoint("/api/tournaments?limit=10", description="Lista de torneos (limitado a 10)")

    # 12. Ranking de jugadores
    test_api_endpoint("/api/rankings/top", description="Top jugadores históricos")

    # 13. Estadísticas generales
    test_api_endpoint("/api/stats/overview", description="Estadísticas generales del dataset")

    # 14. Test con jugador que podría no existir
    test_api_endpoint("/api/player/Juan Perez Inexistente", description="Test con jugador inexistente")

    # 15. Test de predicción con datos inválidos
    invalid_prediction = {
        "player1": "Rafael Nadal",
        "player2": "Rafael Nadal",  # Mismo jugador
        "surface": "Clay"
    }
    test_api_endpoint("/api/predict", method="POST", data=invalid_prediction,
                     description="Test predicción inválida (mismo jugador)")

    print("\n" + "="*80)
    print("🎉 TESTS DE API COMPLETADOS")
    print("="*80)
    print("\n💡 Para iniciar el servidor manualmente:")
    print("   python app.py")
    print("\n🌐 URLs útiles:")
    print(f"   • API Principal: {BASE_URL}")
    print(f"   • Health Check: {BASE_URL}/api/health")
    print(f"   • Jugadores: {BASE_URL}/api/players")
    print(f"   • Predicción: {BASE_URL}/api/predict (POST)")

if __name__ == "__main__":
    main()
