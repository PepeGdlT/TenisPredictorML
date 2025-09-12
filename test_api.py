"""
Test script for the Tennis Prediction API.
Use this to test the API functionality and validate the backend.
"""
import requests
import json
import time
from typing import Dict, Any

class TennisAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def test_connection(self) -> bool:
        """Test if the API server is running."""
        try:
            response = requests.get(f"{self.base_url}/")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        response = requests.get(f"{self.base_url}/status")
        return response.json()

    def list_players(self, limit: int = 10) -> list:
        """Get list of available players."""
        response = requests.get(f"{self.base_url}/players?limit={limit}")
        return response.json()

    def get_player_features(self, player_name: str) -> Dict[str, Any]:
        """Get features for a specific player."""
        response = requests.get(f"{self.base_url}/players/{player_name}/features")
        return response.json()

    def predict_match(self, player1: str, player2: str, surface: str,
                     tournament_info: Dict = None) -> Dict[str, Any]:
        """Make a match prediction."""
        payload = {
            "player1": player1,
            "player2": player2,
            "surface": surface
        }

        if tournament_info:
            payload["tournament_info"] = tournament_info

        response = requests.post(f"{self.base_url}/predict", json=payload)
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        response = requests.get(f"{self.base_url}/model/info")
        return response.json()

    def run_comprehensive_test(self):
        """Run a comprehensive test of all API functionality."""
        print("🎾 Tennis Prediction API - Comprehensive Test")
        print("=" * 50)

        # 1. Test connection
        print("\n1. Testing API connection...")
        if self.test_connection():
            print("✅ API server is running")
        else:
            print("❌ API server is not accessible")
            return

        # 2. Check system status
        print("\n2. Checking system status...")
        try:
            status = self.get_system_status()
            print(f"   Status: {status['status']}")
            print(f"   Data loaded: {status['data_loaded']}")
            print(f"   Model loaded: {status['model_loaded']}")
            print(f"   Available players: {status['available_players']}")
            print(f"   Model type: {status['model_info'].get('model_type', 'Unknown')}")
        except Exception as e:
            print(f"❌ Error getting status: {e}")
            return

        # 3. List available players (solo muestra de ejemplo)
        print("\n3. Getting sample players...")
        try:
            players = self.list_players(10)  # Solo 10 para muestra
            print(f"   Found {len(players)} sample players")
            if players:
                print(f"   Sample players: {players[:5]}")

                # 4. Test player features
                print("\n4. Testing player features...")
                test_player = players[0]
                player_info = self.get_player_features(test_player)
                print(f"   Player: {test_player}")
                print(f"   Features available: {player_info['available']}")
                if player_info['available']:
                    print(f"   Feature count: {player_info.get('feature_count', 'Unknown')}")

        except Exception as e:
            print(f"❌ Error testing players: {e}")
            return

        # 5. Test model info
        print("\n5. Getting model information...")
        try:
            model_info = self.get_model_info()
            print(f"   Model class: {model_info.get('model_class', 'Unknown')}")
            print(f"   Features count: {model_info.get('n_features', 'Unknown')}")
            print(f"   Has imputer: {model_info.get('has_feature_imputer', False)}")
        except Exception as e:
            print(f"❌ Error getting model info: {e}")

        # 6. Test predictions
        print("\n6. Testing match predictions...")

        # Buscar directamente a Alcaraz y Sinner usando get_player_features
        target_players_variations = [
            # Posibles nombres para Alcaraz
            ['Carlos Alcaraz', 'Carlos Alcaraz Garfia', 'Alcaraz', 'C. Alcaraz'],
            # Posibles nombres para Sinner
            ['Jannik Sinner', 'J. Sinner', 'Sinner']
        ]

        available_target_players = []

        print(f"   🔍 Searching for Alcaraz and Sinner using direct API calls...")

        # Buscar Alcaraz
        print(f"   🔍 Searching for Alcaraz...")
        alcaraz_found = False
        for name in target_players_variations[0]:
            try:
                player_info = self.get_player_features(name)
                if player_info['available']:
                    available_target_players.append(name)
                    print(f"   ✅ Found Alcaraz as: {name}")
                    alcaraz_found = True
                    break
            except:
                continue

        if not alcaraz_found:
            print(f"   ❌ Could not find Alcaraz")

        # Buscar Sinner
        print(f"   🔍 Searching for Sinner...")
        sinner_found = False
        for name in target_players_variations[1]:
            try:
                player_info = self.get_player_features(name)
                if player_info['available']:
                    available_target_players.append(name)
                    print(f"   ✅ Found Sinner as: {name}")
                    sinner_found = True
                    break
            except:
                continue

        if not sinner_found:
            print(f"   ❌ Could not find Sinner")

        print(f"   📋 Available target players: {available_target_players}")

        if len(available_target_players) >= 2:
            try:
                player1, player2 = available_target_players[0], available_target_players[1]
                surfaces = ['Hard', 'Clay', 'Grass']

                print(f"   🌟 FEATURED MATCH: {player1} vs {player2}")

                for surface in surfaces:
                    print(f"\n   Testing: {player1} vs {player2} on {surface}")

                    prediction = self.predict_match(player1, player2, surface)

                    if 'player_1_win_probability' in prediction:
                        p1_prob = prediction['player_1_win_probability']
                        p2_prob = prediction['player_2_win_probability']
                        winner = prediction['predicted_winner']
                        confidence = prediction['confidence']

                        print(f"      {player1}: {p1_prob:.1%}")
                        print(f"      {player2}: {p2_prob:.1%}")
                        print(f"      Predicted winner: Player {winner}")
                        print(f"      Confidence: {confidence:.1%}")
                        print(f"      Features used: {prediction['features_used']}")

                        # Mostrar features importantes si están disponibles
                        if 'top_features' in prediction and prediction['top_features']:
                            print(f"      Top 3 features:")
                            for i, (feature, data) in enumerate(list(prediction['top_features'].items())[:3]):
                                importance = data.get('importance', 0)
                                value = data.get('value', 0)
                                print(f"        {i+1}. {feature}: {importance:.3f} (value: {value:.3f})")
                    else:
                        print(f"      ❌ Prediction failed: {prediction}")

                    time.sleep(0.5)  # Small delay between requests

            except Exception as e:
                print(f"❌ Error testing predictions: {e}")

        elif len(players) >= 2:
            # Fallback: usar los primeros dos jugadores disponibles
            try:
                player1, player2 = players[0], players[1]
                surfaces = ['Hard', 'Clay', 'Grass']

                print(f"   ⚠️  Alcaraz/Sinner not found, using: {player1} vs {player2}")

                for surface in surfaces:
                    print(f"\n   Testing: {player1} vs {player2} on {surface}")

                    prediction = self.predict_match(player1, player2, surface)

                    if 'player_1_win_probability' in prediction:
                        p1_prob = prediction['player_1_win_probability']
                        p2_prob = prediction['player_2_win_probability']
                        winner = prediction['predicted_winner']
                        confidence = prediction['confidence']

                        print(f"      {player1}: {p1_prob:.1%}")
                        print(f"      {player2}: {p2_prob:.1%}")
                        print(f"      Predicted winner: Player {winner}")
                        print(f"      Confidence: {confidence:.1%}")
                        print(f"      Features used: {prediction['features_used']}")
                    else:
                        print(f"      ❌ Prediction failed: {prediction}")

                    time.sleep(0.5)  # Small delay between requests

            except Exception as e:
                print(f"❌ Error testing predictions: {e}")
        else:
            print("   ❌ Not enough players available for predictions")

        print("\n" + "=" * 50)
        print("🎾 Test completed!")

def main():
    """Main test function."""
    print("Starting Tennis Prediction API tests...")

    # Wait a moment for server to be ready
    time.sleep(2)

    tester = TennisAPITester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
