import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np

def get_exp_results(log_path: str, field_name: str = 'loss/valid'):

    res = {}

    # Just parse all info from file.
    print("Parsing tf event files.")
    for fname in tqdm(list(os.listdir(log_path))):
        event_res = {}

        # Get the path to the tf events file.
        fpath = os.path.join(log_path, fname)
        for fname_ in os.listdir(fpath):
            if 'events.out' in fname_:
                fpath = os.path.join(fpath, fname_)
                break

        for e in tf.compat.v1.train.summary_iterator(fpath):
            for v in e.summary.value:
                event_res.setdefault(v.tag, []).append((e.wall_time, v.simple_value))

        # Turn (clock time, val) to list of vals in correct order.
        for k, vs in event_res.items():
            event_res[k] = [val for clocktime, val in sorted(vs)]

        res[fname] = event_res

    # Parse which runs are runs w/ same params.
    reran_res = {}
    for run_name, run_res in res.items():
        split_name = run_name.split('_')
        run_num = int(split_name[-1])
        run_name_base = '_'.join(split_name[:-1])
        reran_res.setdefault(run_name_base, {})[run_num] = run_res

    # Determine summary stats.
    print("Calculating summary stats.")
    final_res = {}
    for arch, arch_runs in reran_res.items():
        vals = []
        for run_num, run_results in arch_runs.items():
            try:
                vals.append(run_results[field_name][-1])
            except KeyError:
                print(f'cannot parse run {arch} {run_num}')

        mean = np.mean(vals)
        std = np.std(vals)
        final_res[arch] = (mean, std)

    return final_res


def print_typical(res):

    vals = []
    for exp_name, exp_res in res.items():
        # import
        d = int(exp_name.split('[32, ')[-1].split(']')[0])
        base_arch = exp_name.split('_')[0]
        vals.append(((d, base_arch), exp_res))

    last_d = None
    for ((d, arch_name), (mean, std)) in sorted(vals):
        print(arch_name, d, str(f"{mean:2f} +- {std:2f}"))
        if d == last_d:
            print()

        last_d = d

def print_table(cifar_res, mnist_res):
    """Prints the table formatted for latex."""

    vals = {}
    for exp_name, exp_res in cifar_res.items():
        d = int(exp_name.split('[32, ')[-1].split(']')[0])
        base_arch = exp_name.split('_')[0]
        vals[f"{d}_CIFAR_{base_arch}"] = exp_res

    for exp_name, exp_res in mnist_res.items():
        d = int(exp_name.split('[32, ')[-1].split(']')[0])
        base_arch = exp_name.split('_')[0]
        vals[f"{d}_MNIST_{base_arch}"] = exp_res

    template = """
\\begin{tabular}{ccSS} \\toprule

    {dataset} & {$n$} & \\textbf{VQ-VAE} & \\textbf{hVQ-VAE} \\\\ \midrule

    \multirow{4}{*}{\\rotatebox[origin=c]{90}{MNIST}}
      & 2 & 2_MNIST_vq_m {\scriptsize$\pm$ 2_MNIST_vq_std} & 2_MNIST_hvq_m {\scriptsize$\pm$ 2_MNIST_hvq_std} \\\\
      & 5 & 5_MNIST_vq_m {\scriptsize$\pm$ 5_MNIST_vq_std} & 5_MNIST_hvq_m {\scriptsize$\pm$ 5_MNIST_hvq_std}  \\\\
      & 10 & 10_MNIST_vq_m {\scriptsize$\pm$ 10_MNIST_vq_std} & 10_MNIST_hvq_m {\scriptsize$\pm$ 10_MNIST_hvq_std} \\\\
      & 20 & 20_MNIST_vq_m {\scriptsize$\pm$ 20_MNIST_vq_std} & 20_MNIST_hvq_m {\scriptsize$\pm$ 20_MNIST_hvq_std} \\\\ \midrule

    \multirow{4}{*}{\\rotatebox[origin=c]{90}{CIFAR-10}} 
      & 2 & 2_CIFAR_vq_m {\scriptsize$\pm$ 2_CIFAR_vq_std} & 2_CIFAR_hvq_m {\scriptsize$\pm$ 2_CIFAR_hvq_std} \\\\
      & 5 & 5_CIFAR_vq_m {\scriptsize$\pm$ 5_CIFAR_vq_std} & 5_CIFAR_hvq_m {\scriptsize$\pm$ 5_CIFAR_hvq_std}  \\\\
      & 10 & 10_CIFAR_vq_m {\scriptsize$\pm$ 10_CIFAR_vq_std} & 10_CIFAR_hvq_m {\scriptsize$\pm$ 10_CIFAR_hvq_std} \\\\
      & 20 & 20_CIFAR_vq_m {\scriptsize$\pm$ 20_CIFAR_vq_std} & 20_CIFAR_hvq_m {\scriptsize$\pm$ 20_CIFAR_hvq_std} \\\\ \\bottomrule

\end{tabular}
    """

    for key_name, val in vals.items():
        template = template.replace(key_name + '_m', str(round(val[0], 2)))
        template = template.replace(key_name + '_std', str(round(val[1] * 2, 2)))

    print(template)


if __name__ == '__main__':

    experiments = {
        'CIFAR10 classification':
            {
                'log_path': "saved/cifar-classification/log/1212_004955/",
                'field_name': 'classification_accuracy/valid'
            },
        'MNIST classification':
            {
                'log_path': "saved/mnist-classification/log/1212_035327/",
                'field_name': 'classification_accuracy/valid'
            },
        'CIFAR10 reconstruction':
            {
                'log_path': "saved/cifar-reconstruction/log/1212_101405/",
                'field_name': 'reconstruction_loss/valid'
            },
        'MNIST reconstruction':
            {
                'log_path': "saved/mnist-reconstruction/log/1212_113703/",
                'field_name': 'reconstruction_loss/valid'
            }
    }

    # Classification.
    print("Classification")
    mnist_res = get_exp_results(**experiments['MNIST classification'])
    cifar_res = get_exp_results(**experiments['CIFAR10 classification'])
    print_table(mnist_res=mnist_res, cifar_res=cifar_res)

    # Reconstruction.
    print("Reconstruction")
    mnist_res = get_exp_results(**experiments['MNIST reconstruction'])
    cifar_res = get_exp_results(**experiments['CIFAR10 reconstruction'])
    print_table(mnist_res=mnist_res, cifar_res=cifar_res)

    # for exp_name, exp_info in experiments.items():
    #     res = get_exp_results(**exp_info)
    #     print(exp_name)
    #     print('-----')
    #     print_table(res)
    #     print('-----')
