import os
import sys
import json
from collections import defaultdict
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUTS = os.path.join(BASE_DIR, 'outputs')
PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

H2H_JSON = os.path.join(OUTPUTS, 'h2h_all.json')
SURF_JSON = os.path.join(OUTPUTS, 'surface_elos_by_player.json')

FAMOUS = [
    'Carlos Alcaraz',
    'Rafael Nadal',
    'Jannik Sinner',
    'Novak Djokovic'
]


def load_artifacts():
    """Carga estricta de artefactos desde outputs/.
    No reconstruye desde processed/ bajo ninguna circunstancia.
    """
    missing = []
    if not os.path.exists(H2H_JSON):
        missing.append(H2H_JSON)
    if not os.path.exists(SURF_JSON):
        missing.append(SURF_JSON)
    if missing:
        raise FileNotFoundError(
            "No se encontraron los artefactos requeridos en outputs/. "
            f"Faltan: {missing}. Ejecuta el notebook (sección 20.1) o el script de build para generarlos."
        )
    with open(H2H_JSON, encoding='utf-8') as f:
        h2h = json.load(f)
    with open(SURF_JSON, encoding='utf-8') as f:
        surf = json.load(f)
    return h2h, surf


def print_pair_h2h(h2h, a, b):
    k = '|||'.join(sorted([a, b]))
    rec = h2h['global'].get(k)
    print(f"\n🧮 H2H Global: {a} vs {b}")
    if rec:
        w_a = rec['wins'].get(a, 0)
        w_b = rec['wins'].get(b, 0)
        print(f"   {a}: {w_a}  |  {b}: {w_b}  |  total: {rec['count']}")
    else:
        print('   sin partidos')

    print('   Por superficie:')
    found_any = False
    for s in ['Hard', 'Clay', 'Grass', 'Carpet', 'Unknown']:
        ks = f"{k}|||{s}"
        rs = h2h['by_surface'].get(ks)
        if rs:
            found_any = True
            w_a = rs['wins'].get(a, 0)
            w_b = rs['wins'].get(b, 0)
            print(f"   - {s}: {w_a}-{w_b} (total {rs['count']})")
    if not found_any:
        print('   (sin registros por superficie)')


def print_player_surface_elos(surf_elos, player):
    print(f"\n📈 Surface ELO (último partido por superficie) para {player}:")
    rec = surf_elos.get(player)
    if not rec:
        print('   sin datos')
        return
    for s, v in sorted(rec.items()):
        try:
            print(f"   - {s}: {float(v):.1f}")
        except Exception:
            print(f"   - {s}: {v}")


def main():
    h2h, surf = load_artifacts()

    # CLI simple: 0 args -> demo; 1 arg -> surface ELO jugador; 2 args -> H2D entre 2 jugadores
    args = [a.strip() for a in sys.argv[1:]]
    if len(args) == 2:
        a, b = args
        print_pair_h2h(h2h, a, b)
        print_player_surface_elos(surf, a)
        print_player_surface_elos(surf, b)
        return
    if len(args) == 1:
        print_player_surface_elos(surf, args[0])
        return

    # Demo por defecto
    print('🔍 Demo: H2H entre jugadores famosos y ELO por superficie')
    for i in range(len(FAMOUS)):
        for j in range(i + 1, len(FAMOUS)):
            print_pair_h2h(h2h, FAMOUS[i], FAMOUS[j])

    for p in FAMOUS:
        print_player_surface_elos(surf, p)

    # Checks básicos estilo test
    print('\n✅ Checks rápidos:')
    assert isinstance(h2h, dict) and 'global' in h2h and 'by_surface' in h2h
    assert isinstance(surf, dict)
    # Si hay un registro global, sus conteos deben ser coherentes
    if h2h['global']:
        any_pair = next(iter(h2h['global'].values()))
        total = any_pair['count']
        sum_wins = sum(any_pair['wins'].values())
        assert sum_wins == total, 'La suma de victorias no coincide con el total de partidos'
    print('   Estructuras correctas y conteos coherentes')


if __name__ == '__main__':
    main()
