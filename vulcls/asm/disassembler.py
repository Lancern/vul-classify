import re

import r2pipe


def disassemble_fp(binary_file: str, output_fp):
    # r = r2pipe.open(binary_file)
    #
    # r.cmd('aaaa')
    # funcs = r.cmd('afl').split('\n')
    #
    # for func_name in map(lambda s: s.strip(), funcs):
    #     if len(func_name) == 0:
    #         continue
    #
    #     func_name = func_name.split()
    #     output_fp.write('\n{}:'.format(func_name[-1]))
    #
    #     instructions = r.cmd('pdf @ ' + func_name[0]).split('\n')
    #     for instr in map(lambda s: s.strip(), instructions):
    #         if len(instr) == 0:
    #             continue
    #
    #         if instr[0] not in '|\\':
    #             continue
    #
    #         if len(instr) < 14:
    #             continue
    #
    #         if instr[12:14] == '0x':
    #             if len(instr) < 29:
    #                 continue
    #
    #             if instr[28] not in '0123456789abcdef':
    #                 continue
    #
    #             instr = re.search(r'([^;]*)', instr).group(0)
    #             output_fp.write('\t{}\n'.format(instr))

    r = r2pipe.open(binary_file)

    # analysis
    r.cmd('aaaa')

    func = r.cmd('afl').split('\n')

    for x in func:
        if x == '':
            continue
        y = x.split(' ')
        # print(x)
        # print('\n' + y[-1] + ':')
        output_fp.write('\n' + y[-1] + ':\n')
        raw_str_array = r.cmd('pdf @ ' + y[0]).split('\n')
        for raw_str in raw_str_array:
            if len(raw_str) < 1:
                continue
            if raw_str[0] == '|' or raw_str[0] == '\\':
                pass
            else:
                continue
            if len(raw_str) < 14:
                continue
            if raw_str[12] == '0' and raw_str[13] == 'x':
                if len(raw_str) < 29:
                    continue
                if raw_str[28] in '0123456789abcdef':
                    pass
                else:
                    continue
                s = raw_str[43:]
                s = re.search(r'([^;]*)', s).group(0)
                # print('write:' + s)
                output_fp.write('\t' + s + '\n')


def disassemble(binary_file: str, output_file: str):
    with open(output_file, 'w') as output_fp:
        disassemble_fp(binary_file, output_fp)


__all__ = ['disassemble', 'disassemble_fp']
