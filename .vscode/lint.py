import re
import sys
import subprocess

try:
    subprocess.check_output([
        'verilator', 
        '--lint-only',
        '-f',
        './.vscode/linter.vc',
        '--error-limit',
        '9999',
        '-Wall',
        '-Wno-MULTITOP',
        '-Wno-MODDUP'
    ], stderr=subprocess.STDOUT)
except subprocess.CalledProcessError as e:
    out = e.output.decode('ascii')

out = re.sub(r'%.*\/\.bender\/[^%]*', '', out)
# print(out)

if len(sys.argv) > 1:
    pat = f'%.*{sys.argv[1]}[^%]*'
    out = re.findall(pat, out)
    out = '\n'.join(out)

print(out, file=sys.stderr)