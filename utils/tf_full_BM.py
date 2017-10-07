#!/usr/bin/python

# build eigen
# build tensorflow
# install tensorflow

# run eigen benchmarks
# run dlbench
# run tensorflow benchmarks

import os
import shutil
import subprocess
import datetime
import time
import itertools
import socket
import argparse


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Repo(object):
    def __init__(self, url, branch, workspace):
        self.URL = url
        self.BRANCH = branch
        self.DIRNAME = url.split("/")[-1].split(".")[0]
        self.WORKSPACE = workspace

    def pull_git(self):
        cmd = "cd " + self.WORKSPACE
        if not os.path.exists(self.WORKSPACE + "/" + self.DIRNAME):
            cmd += " && git clone " + self.URL
        cmd += " && cd " + self.DIRNAME + " && git checkout " + self.BRANCH
        print bcolors.WARNING + cmd + bcolors.ENDC
        os.system(cmd)
        print bcolors.OKBLUE + "DONE!" + bcolors.ENDC

    def pull_hg(self):
        cmd = "cd " + self.WORKSPACE
        if not os.path.exists(self.WORKSPACE + "/" + self.DIRNAME):
            cmd += " && hg clone " + self.URL
        cmd += " && cd " + self.DIRNAME + " && hg up " + self.BRANCH + " && hg pull"
        print bcolors.WARNING + cmd + bcolors.ENDC
        os.system(cmd)
        print bcolors.OKBLUE + "DONE!" + bcolors.ENDC

class Workspace(object):
    def __init__(self, computecpp_root, workspace, eigen_branch, tf_branch,
                 dlbench_branch, benchmarks_branch, package):
        self.now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        self.workspace = workspace
        self.mkdir_and_go(self.workspace)
        self.tmp = self.mkdir("tmp")
        self.log = self.mkdir("log")
        self.log = self.mkdir("log/" + self.now)
        self.computecpp = self.fetch(
            "http://computecpp.codeplay.com/downloads/computecpp-ce/latest/"+package+".tar.gz",
            workspace=self.tmp, directory_="computecpp", file_path=package+".tar.gz")
        self.tensorflow = Repo("https://github.com/lukeiwanski/tensorflow.git", tf_branch, self.tmp)
        self.benchmarks = Repo("https://github.com/tensorflow/benchmarks.git",
                               benchmarks_branch, self.tmp)
        self.dlbench = Repo("https://github.com/tfboyd/dlbench.git",
                            dlbench_branch, self.tmp)
        self.eigen = Repo("https://bitbucket.org/mehdi_goli/opencl", eigen_branch, self.tmp)
        self.ip = socket.gethostbyname(socket.gethostname())

    def setup(self):
        self.tensorflow.pull_git()
        self.benchmarks.pull_git()
        self.dlbench.pull_git()
        self.eigen.pull_hg()

    def mkdir(self, path, delete = False):
        if os.path.exists(path):
            if delete:
                print path + " exists cleaning it..."
                shutil.rmtree(path)
                print bcolors.OKBLUE + "DONE!" + bcolors.ENDC
                os.mkdir(path)
        else:
            os.mkdir(path)
        return self.workspace + "/" + path

    def fetch(self, url, workspace, directory_, file_path, delete = False):
        print "Fetching "+ url
        path = workspace+"/"+directory_
        cmd = "cd " + workspace

        # fetch tar.gz
        if not os.path.exists(workspace + "/" + file_path):
            cmd += " && wget " + url

        # make temp
        if not os.path.exists(path + "_tmp"):
            os.mkdir(path + "_tmp")
            cmd += " && tar xf " + file_path + " -C " + directory_ + "_tmp"

        # do we clean?
        if delete:
            if os.path.exists(path):
                print path + " exists cleaning it..."
                shutil.rmtree(path)
                print bcolors.OKBLUE + "DONE!" + bcolors.ENDC
                #full cmd
                full_cmd = "cd " + workspace + " && wget " + url +  " && tar xf " + file_path + " -C " + directory_ + "_tmp"
                print bcolors.WARNING + full_cmd + bcolors.ENDC
                os.system(full_cmd)
        else:
            # we have final file
            if os.path.exists(path):
                # remove _tmp
                if os.path.exists(path + "_tmp"):
                    os.system("rm -rf " + path + "_tmp")
                return path
            else:
                # create _tmp
                print bcolors.WARNING + cmd + bcolors.ENDC
                os.system(cmd)

        # mv *_tmp to dir
        cmd = "mv " + path + "_tmp/* " + path + " && rm -rf " + path + "_tmp" + " && chmod +x " + path+"/bin/*"
        print bcolors.WARNING + cmd + bcolors.ENDC
        os.system(cmd)
        print bcolors.OKBLUE + "DONE!" + bcolors.ENDC
        return path


    def mkdir_and_go(self, path):
        self.mkdir(path)
        os.chdir(path)

    def execute(self, cmd, log_modifier, cwd = "", my_env = []):
        print bcolors.WARNING + cmd + bcolors.ENDC
        p = subprocess.Popen(cmd, env=my_env, cwd=cwd, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode:
            log_file = self.log+"/"+self.now+"-FAIL-" + log_modifier + ".log"
            file_ = open(log_file, "w")
            print bcolors.FAIL + "FAIL: " + bcolors.ENDC + log_modifier + " " + log_file
            file_.write(err)
        else:
            log_file = self.log+"/"+self.now+"-PASS-" + log_modifier + ".log"
            file_ = open(log_file, "w")

            print bcolors.OKBLUE + "PASS: " + bcolors.ENDC + log_modifier + " " + log_file
            file_.write(out)
        file_.close()
        return p.returncode

    def build_bench_eigen(self):
        print "Compiling and Running Eigen Benchmarks"
        cwd = self.tmp + "/" + self.eigen.DIRNAME+"/bench/tensors"
        my_env = os.environ
        my_env["COMPUTECPP_PACKAGE_ROOT_DIR"] = self.computecpp

        cmd = "bash eigen_sycl_bench.sh"
        self.execute(cmd=cmd, log_modifier="eigen", cwd=cwd, my_env=my_env)
        print bcolors.OKBLUE + "DONE!" + bcolors.ENDC

    def build_install_tf(self):
        print "Compiling and Installing TF"
        cwd = self.tmp + "/" + self.tensorflow.DIRNAME
        my_env = os.environ
        my_env["TF_NEED_OPENCL"] = "1"
        my_env["HOST_CXX_COMPILER"] = "/usr/bin/g++"
        my_env["HOST_C_COMPILER"] = "/usr/bin/gcc"
        my_env["COMPUTECPP_TOOLKIT_PATH"] = self.computecpp

        cmd = "bash configure yes"
        ret = self.execute(cmd=cmd, log_modifier="tf_configure", cwd=cwd, my_env=my_env)

        cmd = "bazel build -c opt --copt=-msse4.1 --copt=-msse4.1 --copt=-mavx --copt=-mavx2 --copt=-mfma --copt -Wno-unused-command-line-argument --copt -Wno-duplicate-decl-specifier --config=sycl //tensorflow/tools/pip_package:build_pip_package"
        ret &= self.execute(cmd=cmd, log_modifier="tf_build", cwd=cwd, my_env=my_env)

        if not ret:
            cmd = "bazel-bin/tensorflow/tools/pip_package/build_pip_package " + self.workspace + "/" + self.tmp + "/tensorflow_pkg"
            self.execute(cmd=cmd, log_modifier="tf_install", cwd=cwd, my_env=my_env)

            # dont bother when not installed in the first place
            cmd = "pip uninstall tensorflow -y"
            self.execute(cmd=cmd, log_modifier="tf_uninstall_pip", cwd=cwd, my_env=my_env)

            cmd = "pip install --user " + self.workspace + "/" + self.tmp + "/tensorflow_pkg/tensorflow-*.whl"
            ret = self.execute(cmd=cmd, log_modifier="tf_install_pip", cwd=cwd, my_env=my_env)
            if ret:
                print bcolors.FAIL + "FAIL!" + bcolors.ENDC
                return

        else:
            print bcolors.FAIL + "FAIL!" + bcolors.ENDC
            return

        print bcolors.OKBLUE + "DONE!" + bcolors.ENDC

    def run_benchmarks(self):
        print "Running TF benchmarks"
        cwd = self.tmp + "/" + self.benchmarks.DIRNAME+"/scripts/tf_cnn_benchmarks"
        my_env = os.environ
        my_env["LD_LIBRARY_PATH"] = self.computecpp+"/lib"
        my_env["TF_CPP_MIN_LOG_LEVEL"] = "2"

        models = ["alexnet", "vgg16", "resnet50", "inception3"]
        # using gpu device is more flexible here - node that suppose to be
        # accelerated on GPU will use SYCL device when aviable
        targets = ["cpu", "gpu"]
        for model, target in itertools.product(models, targets):
            file_name = self.ip+"_"+model+"_inference_"+target
            # inference
            cmd = "python tf_cnn_benchmarks.py --num_batches=10 --device="+target+" --batch_size=1 --forward_only=true --model="+model+" --data_format=NHWC --trace_file="+self.now+"_"+file_name+".js"
            self.execute(cmd=cmd, log_modifier=file_name, cwd=cwd, my_env=my_env)

        for model, target in itertools.product(models, targets):
            file_name = self.ip+"_"+model+"_training_"+target
            # inference
            cmd = "python tf_cnn_benchmarks.py --batch_size=32 --num_batches=10 --device="+target+" --model="+model+" --data_format=NHWC --trace_file="+self.now+"_"+file_name+".js"
            self.execute(cmd=cmd, log_modifier=file_name, cwd=cwd, my_env=my_env)

    def run_dlbench(self):
        print "Running dlbench"
        cwd = self.tmp + "/" + self.dlbench.DIRNAME+"/tools/tensorflow/fc"
        my_env = os.environ
        my_env["LD_LIBRARY_PATH"] = self.computecpp+"/lib"
        my_env["TF_CPP_MIN_LOG_LEVEL"] = "2"

        cmd = "python fcn5_mnist.py --use_dataset=False --data_dir="+ cwd + "/mnist_datasets"
        self.execute(cmd=cmd, log_modifier="dlbench", cwd=cwd, my_env=my_env)



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--computecpp", dest="computecpp", default="/usr/local/computecpp",
                        help="Path to ComputeCpp root")
    parser.add_argument("-p", "--package", dest="package", default="Ubuntu-16.04-64bit",
                        help="Version of ComputeCpp to use")
    parser.add_argument("-w", "--workspace", dest="workspace", default=os.getenv("HOME")+"/autogen_workspace",
                        help="Where to create workspace")
    parser.add_argument("-e", "--eigen_branch", dest="eigen", default="Eigen-OpenCL-Optimised",
                        help="Eigen branch to use")
    parser.add_argument("-t", "--tf_branch", dest="tf", default="dev/amd_gpu",
                        help="TensorFlow branch to use")
    parser.add_argument("-b", "--benchmarks_branch", dest="benchmarks", default="master",
                        help="TensorFlow Benchmarks branch to use")
    parser.add_argument("-d", "--blbench_branch", dest="dlbench", default="data_sets",
                        help="DLBench branch to use")


    args = parser.parse_args()
    print(args)

    workspace = Workspace(computecpp_root=args.computecpp,
                          workspace=args.workspace, eigen_branch=args.eigen,
                          tf_branch=args.tf, benchmarks_branch=args.benchmarks,
                          dlbench_branch=args.dlbench, package=args.package)
    workspace.setup()
    workspace.build_bench_eigen()
    workspace.build_install_tf()
    workspace.run_benchmarks()
    workspace.run_dlbench()

if __name__ == "__main__":
    main()
