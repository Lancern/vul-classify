import re

import r2pipe


def disassemble(binary_file: str, output_file: str):
    r = r2pipe.open(binary_file)

    r.cmd('aaaa')
    funcs = r.cmd('afl').split('\n')

    with open(output_file, 'w') as output_fp:
        for func_name in map(lambda s: s.strip(), funcs):
            if len(func_name) == 0:
                continue

            func_name = func_name.split()
            output_fp.write('\n{}:'.format(func_name[-1]))

            instructions = r.cmd('pdf @ ' + func_name[0]).split('\n')
            for instr in map(lambda s: s.strip(), instructions):
                if len(instr) == 0:
                    continue

                if instr[0] not in '|\\':
                    continue

                if len(instr) < 14:
                    continue

                if instr[12:14] == '0x':
                    if len(instr) < 29:
                        continue

                    if instr[28] not in '0123456789abcdef':
                        continue

                    instr = re.search(r'([^;]*)', instr).group(0)
                    output_fp.write('\t{}\n'.format(instr))


__all__ = ['disassemble']
