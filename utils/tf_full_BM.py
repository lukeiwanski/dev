#!/usr/bin/env python

import os
import shutil
import subprocess
import datetime
import time
import itertools
import socket
import argparse
import sys
import tarfile
import io

import shutil
import requests
import csv
import glob
import numpy as np
import pandas as pd

COMPUTECPP_BASE_URL_TMPL = \
        "https://computecpp.codeplay.com/downloads/computecpp-ce/latest/{package_ver}.tar.gz"
TF_UPSTREAM_REPO = "https://github.com/lukeiwanski/tensorflow.git"
TF_BENCHMARKS_REPO = "https://github.com/tensorflow/benchmarks.git"
HTML_VIEW_TMPL = \
"""
<link href='https://cdnjs.cloudflare.com/ajax/libs/c3/0.4.18/c3.min.css' rel='stylesheet'>
    <script src='https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.js' charset='utf-8'></script>
    <script src='https://cdnjs.cloudflare.com/ajax/libs/c3/0.4.18/c3.js'></script>
    <div id='chart'></div>
    <script>
    var chart = c3.generate({data:{x:'version',url:'%(CSV_FILE_PATH)s',type:'scatter'},tooltip:{grouped:false},bindto:'#chart',axis:{y:{show:true,max:100,min:0,ticks:5,padding:{top:1,bottom:0},},x:{type:'category'}}});
    </script>
"""

class Colors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Repo(object):
    """
    Utilities for handling git repos
    """
    def __init__(self, url, branch, workspace):
        self.url = url
        self.branch = branch
        self.dirname = url.split("/")[-1].split(".")[0]
        self.workspace = workspace
        self.hash = branch

    def pull_git(self):
        """update the repository to default remote HEAD and get the new sha1 ref"""
        repo_dir = os.path.join(self.workspace, self.dirname)
        if not os.path.exists(repo_dir):
            subprocess.check_call(['git', 'clone', self.url, os.path.abspath(repo_dir)])

        cmds = [
            ['git', 'fetch'],
            ['git', 'reset', '--hard', 'HEAD'],
            ['git', 'clean', '-dfx'],
            ['git', 'pull', 'origin', self.branch],
        ]
        for cmd in cmds:
            print(Colors.WARNING + ' ' +  ' '.join(cmd) + Colors.ENDC)
            subprocess.check_call(cmd, cwd=repo_dir)
            print(Colors.OKBLUE + "DONE!" + Colors.ENDC)

        self.hash = subprocess.check_output(
            "git rev-parse --verify HEAD".split(),
            cwd=repo_dir
        ).rstrip()

    def hash(self):
        return self.hash

class Workspace(object):
    def __init__(self, computecpp_root, workspace, tf_branch, benchmarks_branch,
                 package, report, tf_build_options, webpage, save_traces):
        now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
        self.tf_build_options = tf_build_options
        self.report = report
        self.workspace = workspace
        self.webpage = webpage
        self.mkdir_and_go(self.workspace)

        self.tmp = self.mkdir("tmp")
        self.log = self.mkdir("log")
        self.log = self.mkdir(os.path.join("log", now))
        self.save_traces = save_traces

        if not os.path.exists(computecpp_root):
            url = COMPUTECPP_BASE_URL_TMPL.format(package_ver=package)
            self.computecpp = self.fetch(
                url,
                workspace=self.tmp,
                directory="computecpp",
            )
            self.computecpp = os.path.join(self.computecpp, os.listdir(self.computecpp)[0])
        else:
            self.computecpp = computecpp_root

        self.tensorflow = Repo(TF_UPSTREAM_REPO, tf_branch, self.tmp)
        self.benchmarks = Repo(TF_BENCHMARKS_REPO, benchmarks_branch, self.tmp)
        # FIXME
        self.ip = socket.gethostbyname(socket.gethostname())
        self.tf_branch = tf_branch

    def gen_version(self, ver_string, rep):
        """
        generate a triple of hashes uniquely identifying computecpp version
        (clang, llvm)
        """
        ret = ver_string.split(") (")
        clang = ret[0].split(" ")[-1]
        llvm = ret[1].split(" ")[-1]
        return "%s-%s-%s" % (rep[:7], clang[:7], llvm[:7])

    def setup(self):
        self.tensorflow.pull_git()
        self.benchmarks.pull_git()
        cwd = self.tmp
        my_env = os.environ

        ver_cmd = [os.path.join(self.computecpp, 'bin', 'compute++'), "--version"]
        ok, compute_cpp_ver_string = self.execute(
            ver_cmd,
            log_modifier="compiler_version_gen",
            cwd=cwd,
            my_env=my_env
        )

        self.workspace_version = self.gen_version(compute_cpp_ver_string, self.tensorflow.hash)

    def mkdir(self, path, delete = False):
        if os.path.exists(path):
            if delete:
                print(path + " exists cleaning it...")
                shutil.rmtree(path)
                print(Colors.OKBLUE + "DONE!" + Colors.ENDC)
                os.mkdir(path)
        else:
            os.mkdir(path)
        return self.workspace + "/" + path

    def mkdir_and_go(self, path):
        self.mkdir(path)
        os.chdir(path)

    def execute(self, cmd, log_modifier, cwd=None, my_env=None):
        cwd = cwd or os.getcwd()
        my_env = my_env or []
        print(Colors.WARNING + ' '.join(cmd) + Colors.ENDC)

        fail_log_file_path = os.path.join(self.log,  "FAIL-" + log_modifier + ".log")
        pass_log_file_path = os.path.join(self.log, "PASS-" + log_modifier + ".log")

        with open(pass_log_file_path, "w") as pass_log_file, \
            open(fail_log_file_path, "w") as fail_log_file:
            p = subprocess.Popen(
                cmd,
                env=my_env,
                cwd=cwd,
                shell=False,
                stdout=pass_log_file,
                stderr=fail_log_file,
                stdin=subprocess.PIPE
            )
            p.communicate()

        with open(pass_log_file_path, 'r') as content_file:
            content = content_file.read()

        if p.returncode == 0:
            print(Colors.OKBLUE + "PASS: " + Colors.ENDC + log_modifier + " " + pass_log_file_path)
        else:
            print(Colors.FAIL + "FAIL: " + Colors.ENDC + log_modifier + " " + fail_log_file_path)

        return p.returncode, content

    def fetch(self, url, workspace, directory, delete=False):
        """
        download and install a local copy of computecpp from ``url``, and
        unpack to ``file_path``.
        """
        if delete:
            if os.path.exists(path):
                print(path + " exists cleaning it...")
                shutil.rmtree(path)
                print(Colors.OKBLUE + "DONE!" + Colors.ENDC)

        print("Fetching %r" % url)
        dest_path = os.path.join(workspace, directory)
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with tarfile.open(fileobj=io.BytesIO(response.raw.read()), mode="r:*") as archive:
            archive.extractall(dest_path)

        return dest_path

    def build_install_tf(self):
        print("Compiling and Installing TF")
        cwd = os.path.join(self.tmp, self.tensorflow.dirname)
        my_env = os.environ.copy()
        my_env.update({
            "TF_NEED_OPENCL_SYCL": "1",
            "TF_NEED_COMPUTECPP": "1",
            "HOST_CXX_COMPILER": "/usr/bin/g++",
            "HOST_C_COMPILER": "/usr/bin/gcc",
            "COMPUTECPP_TOOLKIT_PATH": self.computecpp,
        })

        cmd = ["bash", "configure", "yes"]
        ret = self.execute(cmd=cmd, log_modifier="tf_configure", cwd=cwd, my_env=my_env)[0]

        cmd = (
            "bazel build -c opt --copt=-msse4.1 --copt=-msse4.1 --copt=-mavx "
            "--copt=-mavx2 --copt=-mfma --copt -Wno-unused-command-line-argument "
            "--copt -Wno-duplicate-decl-specifier --config=sycl "
            + self.tf_build_options +
            "//tensorflow/tools/pip_package:build_pip_package"
        ).split()

        ret &= self.execute(cmd=cmd, log_modifier="tf_build", cwd=cwd, my_env=my_env)[0]

        if not ret:
            cmd = [
                "bazel-bin/tensorflow/tools/pip_package/build_pip_package",
                os.path.join(self.workspace, self.tmp, "tensorflow_pkg"),
            ]
            self.execute(cmd=cmd, log_modifier="tf_install", cwd=cwd, my_env=my_env)

            # don't bother when not installed in the first place
            cmd = ["pip", "uninstall", "tensorflow", "-y"]
            self.execute(cmd=cmd, log_modifier="tf_uninstall_pip", cwd=cwd, my_env=my_env)

            cmd = ["pip", "install", "--user"] + glob.glob(
                os.path.join(self.tmp, "tensorflow_pkg/tensorflow-*.whl")
            )
            ret = self.execute(cmd=cmd, log_modifier="tf_install_pip", cwd=cwd, my_env=my_env)[0]
            if ret:
                print(Colors.FAIL + "FAIL!" + Colors.ENDC)
                sys.exit(1)

        else:
            print(Colors.FAIL + "FAIL!" + Colors.ENDC)
            sys.exit(1)

        print(Colors.OKBLUE + "DONE!" + Colors.ENDC)

    def run_benchmarks(self):
        print("Running TF benchmarks")
        cwd = os.path.join(self.tmp, self.benchmarks.dirname, 'scripts', 'tf_cnn_benchmarks')
        my_env = os.environ.copy()
        my_env.update({
            "LD_LIBRARY_PATH": os.path.join(self.computecpp, "lib"),
            "TF_CPP_MIN_LOG_LEVEL": "2",
        })

        models = ["trivial", "alexnet", "vgg16", "resnet50", "inception3"]
        # using gpu device is more flexible here - node that suppose to be
        # accelerated on GPU will use SYCL device when aviable
        targets = ["cpu", "gpu"]
        for model, target in itertools.product(models, targets):
            file_name = "%s_%s_inference_%s" % (self.ip, model, target)
            # inference
            n_benches = 10
            cmd = [
                "python", "tf_cnn_benchmarks.py",
                "--num_batches=" + str(n_benches),
                "--device=" + target,
                "--batch_size=1",
                "--forward_only=true",
                "--model=" + model,
                "--data_format=NHWC"
            ]
            if self.save_traces:
                cmd += [
                    "--trace_file=" + os.path.join(
                        self.tmp, self.workspace_version +"_"+ file_name + ".json"
                    )
                ]

            self.execute(cmd=cmd, log_modifier=file_name, cwd=cwd, my_env=my_env)

        for model, target in itertools.product(models, targets):
            file_name = self.ip+"_"+model+"_training_"+target
            # training
            cmd = [
                "python", "tf_cnn_benchmarks.py",
                "--batch_size=32",
                "--num_batches=10",
                "--device=" + target,
                "--model=" + model,
                "--data_format=NHWC",
            ]
            if self.save_traces:
                cmd += [
                    "--trace_file=" + os.path.join(
                        self.tmp, self.workspace_version +"_"+ file_name + ".json"
                    )
                ]

            self.execute(cmd=cmd, log_modifier=file_name, cwd=cwd, my_env=my_env)

    def gen_csv_based_on_log(self):
        file_name = self.report
        file_exists = os.path.exists(file_name)
        with open(file_name, 'a+') as f:
            writer = csv.writer(f)
            header = ["version"]
            path = os.path.join(self.log, "*PASS*.log")
            row = [self.workspace_version]
            for filename in glob.glob(path):
                with open(filename) as f:
                    col_name = ""
                    imgs_sec = []
                    for line in f:
                        if "Model:" in line:
                            col_name += line.split(':')[-1].strip(" ").rstrip()
                        if "Mode:" in line:
                            col_name += "_" + line.split(':')[-1].strip(" ").rstrip()
                        if "Batch size:" in line:
                            col_name += "_" + line.split(':')[-1].rstrip().strip(" global")
                        if "Devices:" in line:
                            col_name += "_" + line.split('/')[-1].split(':')[0].strip(" ").rstrip()
                        # sometimes total images / s is reported as 0
                        # therefore we get all images/sec and choosing the best
                        if "images/sec:" in line:
                            x = line.split(':')[-1].split(' ')[1].rstrip()
                            imgs_sec.append(x)

                    if col_name and imgs_sec:
                        header.append(col_name)
                    if imgs_sec:
                        row.append(np.amax(np.array(imgs_sec).astype(np.float)))

            if not file_exists:
                writer.writerows([header])
            if row:
                writer.writerows([row])

    def gen_webpage(self):
        filename = self.report
        code = HTML_VIEW_TMPL % {'CSV_FILE_PATH': self.report}

        # version
        a = [np.genfromtxt(filename, usecols=[0], delimiter=',', dtype=np.unicode_)]

        # sorted columns
        b = sorted(
            np.transpose(np.genfromtxt(filename, delimiter=',', dtype=np.unicode_))[1:],
            key=lambda tup: tup[0]
        )
        sort = b[:]

        for x in range((len(b)) / 2):
            m1 = sort[2 * x][1:]
            max1 = np.amax(np.asarray(m1, dtype=np.float32)) * 2

            m2 = sort[2 * x + 1][1:]
            max2 = np.amax(np.asarray(m2, dtype=np.float32)) * 2
            max1 = np.maximum(max1, max2)

            out = np.transpose(np.concatenate(([a[0]], [b[2 * x]], [b[2 * x + 1]]), axis=0))
            df = pd.DataFrame(out)
            df.to_csv(str(x) + ".csv", header=None, index=False)
            code += "<div id='c"+str(x)+"'></div><script>var chart = c3.generate({data:{x:'version',url:'"+str(x)+".csv',type:'line'},tooltip:{grouped:false},bindto:'#c"+str(x)+"',axis:{y: { show: true, max:"+str(max1)+", min:0, ticks : 5,padding: {top:1, bottom:0},}, x:{type:'category'}}});</script>"

        file_exists = os.path.exists(self.webpage)
        if not file_exists:
            with open(self.webpage, "w") as f:
                f.write(code)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-c", "--computecpp",
        dest="computecpp",
        default=os.getenv("HOME")+"/autogen_workspace"+"/tmp/computecpp",
        help="Path to ComputeCpp root"
    )
    parser.add_argument(
        "-p", "--package",
        dest="package",
        default="Ubuntu-16.04-64bit",
        help="Version of ComputeCpp to download"
    )
    parser.add_argument(
        "-w", "--workspace",
        dest="workspace",
        default=os.getenv("HOME")+"/autogen_workspace",
       help="Where to create workspace"
    )
    parser.add_argument(
        "-t", "--tf_branch",
        dest="tf",
        default="dev/amd_gpu",
        help="TensorFlow branch to use"
    )
    parser.add_argument(
        "-b", "--benchmarks_branch",
        dest="benchmarks",
        default="master",
        help="TensorFlow Benchmarks branch to use"
    )
    parser.add_argument(
        "-r", "--csv",
        dest="report",
        default="report.csv",
        help="CSV report file name"
    )
    parser.add_argument(
        "-l", "--html",
        dest="webpage",
        default="index.html",
        help="Generated WebPage name"
    )
    parser.add_argument(
        "-x", "--tf_build_options",
        dest="tf_build_options",
        default=" ",
        help="Additional options that will be passed to TF bazel build command"
    )
    parser.add_argument(
        "-s", "--save_traces",
        dest="save_traces",
        default=False,
        help="If set to true traces will be generated"
    )

    args = parser.parse_args()

    workspace = Workspace(
        computecpp_root=args.computecpp,
        workspace=args.workspace,
        tf_branch=args.tf,
        benchmarks_branch=args.benchmarks,
        package=args.package,
        report=args.report,
        tf_build_options=args.tf_build_options,
        webpage=args.webpage,
        save_traces = args.save_traces,
    )
    workspace.setup()
    workspace.build_install_tf()
    workspace.run_benchmarks()
    workspace.gen_csv_based_on_log()
    workspace.gen_webpage()

if __name__ == "__main__":
    main()
