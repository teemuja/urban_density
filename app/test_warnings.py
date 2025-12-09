import sys

# Check the key deprecation fixes
with open('/workspaces/urban_density/app/app.py', 'r') as f:
    content = f.read()

tests = {
    'No unary_union': 'unary_union' not in content,
    'union_all() used': 'union_all()' in content,
    'No choropleth_mapbox': 'choropleth_mapbox' not in content,
    'choropleth_map used': 'choropleth_map' in content,
    'CRS in focus_gdf': 'crs=buildings.crs' in content,
    'morphological_tessellation fixed': 'morphological_tessellation(gdf, clip=' in content,
}

failed = []
for test_name, result in tests.items():
    status = '✓' if result else '✗'
    print(f"{status} {test_name}")
    if not result:
        failed.append(test_name)

if failed:
    print(f"\n❌ {len(failed)} test(s) failed:")
    for f in failed:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\n✓ All critical deprecations and fixes verified!")
