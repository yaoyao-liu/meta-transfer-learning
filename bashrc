# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

# # load additional bin
# # export PATH=:/BS/sun_project/work/software/Anaconda/bin:$PATH
# export PATH=:/usr/bin:$PATH
# export PATH=:/BS/sun_project/work/software/hdf5-1.8.17/build/bin:$PATH
# export PATH=:/BS/3d_deep_learning/work/yangTool/mdb/libraries/liblmdb:$PATH
# export PATH=:/BS/sun_project/work/software/protobuf/build/bin:$PATH
# export PATH=:/BS/sun_project/work/software/gflags-2.0/build/bin:$PATH
# export PATH=:/BS/sun_project/work/software/glog-0.3.3/build/lib:$PATH

# # export PATH=:/usr/lib/nvidia-cuda-toolkit/bin:$PATH
# # export PATH=:/BS/sun_project/work/software/caffe/build/bin:$PATH
# # export PATH=:/BS/3d_deep_learning/work/pcl-install/bin:$PATH
# export PATH=:/BS/sun_project/work/software/libfreenect/build/bin:$PATH
# # export PATH=:/BS/3d_deep_learning/work/pcl/bin:$PATH
# export PATH=:/BS/sun_project/work/software/lua-5.3.3/install/bin:$PATH
# # export PATH=:/BS/sun_project/work/software/opencv-2.4.9/build/bin:$PATH
# export PATH=:/BS/sun_project/work/software/yasm/build/bin:$PATH
# export PATH=:/BS/sun_project/work/software/ffmpeg-3.1.1/build/bin:$PATH
# export PATH=:/BS/sun_project/work/software/openssl-1.0.1t/build/bin:$PATH
# export PATH=:/BS/sun_project/work/software/cnpy/bin:$PATH

# export CAFFE_ROOT=/BS/sun_project/work/software/caffe
# export PATH=:$CAFFE_ROOT/build/tools:$PATH

# # load addtional library
# # export LD_LIBRARY_PATH=:/BS/sun_project/work/software/Anaconda/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/usr/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/hdf5-1.8.17/build/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/3d_deep_learning/work/yangTool/mdb/libraries/liblmdb:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/protobuf/build/lib:$LD_LIBRARY_PATH
# #export LD_LIBRARY_PATH=:/BS/sun_project/work/software/protobuf-2.5.0/build/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/snappy-1.1.1/build/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/gflags-2.0/build/lib:$LD_LIBRARY_PATH
# # export LD_LIBRARY_PATH=:/BS/sun_project/work/software/glog-master/build/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/glog-0.3.3/build/lib:$LD_LIBRARY_PATH
# #export LD_LIBRARY_PATH=:/BS/sun_project/work/software/leveldb:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/usr/lib/nvidia-cuda-toolkit/lib:$LD_LIBRARY_PATH
# # export LD_LIBRARY_PATH=:/BS/sun_project/work/software/caffe/build/lib:$LD_LIBRARY_PATH
# # export LD_LIBRARY_PATH=:/BS/3d_deep_learning/work/pcl-install/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/lib/libpq.so.5:$LD_LIBRARY_PATH
# # export LD_LIBRARY_PATH=:/BS/3d_deep_learning/work/pcl/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/libfreenect/build/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/lua-5.3.3/install/lib:$LD_LIBRARY_PATH
# # export LD_LIBRARY_PATH=:/BS/sun_project/work/software/opencv-2.4.9/build/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/yasm/build/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/ffmpeg-3.1.1/build/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/openssl-1.0.1t/build/lib:$LD_LIBRARY_PATH
# # export LD_LIBRARY_PATH=:/BS/3d_deep_learning/work/PeopleDetection/apollocaffe/build/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# # cudnn-v5.1-rc used for caffe version 1.0.0-rc3
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/cudnn-v5.1-rc/lib64:$LD_LIBRARY_PATH
# # cudnn-v4.0 used for caffe in SituationCrf
# # export LD_LIBRARY_PATH=:/BS/sun_project/work/software/cudnn-v4.0/lib64:$LD_LIBRARY_PATH
# # cudnn-v3 used for caffe-fast-rcnn
# # export LD_LIBRARY_PATH=:/BS/opt/cudnn_v3/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=:/BS/sun_project/work/software/cnpy/lib:$LD_LIBRARY_PATH

# # if [ $HOSTNAME != "menorca" ]; then
# #     export PATH=:/BS/sun_project/work/software/Anaconda/bin:$PATH
# #     export LD_LIBRARY_PATH=:/BS/sun_project/work/software/Anaconda/lib:$LD_LIBRARY_PATH
# # fi

# # export LD_PRELOAD=:/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD

# # export PKG_CONFIG_PATH=/BS/sun_project/work/software/Anaconda/lib/pkgconfig:$PKG_CONFIG_PATH

# # export PYTHONPATH=/BS/sun_project/work/software/caffe/python:$PYTHONPATH


# ###########################
# export LD_LIBRARY_PATH=:/usr/lib/x86_64-linux-gnu:/BS/sun_project/work/software/cudnn-8.0/lib64:/BS/sun_project/work/software/protobuf-3.0.0/build/lib:$LD_LIBRARY_PATH

# # for includes
# INCLUDE_DIR=:/usr/local/include:/usr/include:/usr/include/linux:/BS/sun_project/work/software/cudnn-8.0/include
# #INCLUDE_DIR=:$INCLUDE_DIR:/usr/include/atlas:/BS/sun_project/work/software/protobuf-3.0.0/build/include:/BS/sun_project/work/software/protobuf-3.0.0/build/include/google/protobuf
# export CPLUS_INCLUDE="$CPLUS_INCLUDE:$INCLUDE_DIR"
# export C_INCLUDE="$C_INCLUDE:$INCLUDE_DIR"
# HDF5_DISABLE_VERSION_CHECK=2
# ###################################






#########################################################################################################
##############################################      New    ##############################################
#########################################################################################################

export PATH=:/BS/sun_project/work/software/Anaconda/bin:$PATH
export LD_LIBRARY_PATH=:/BS/sun_project/work/software/Anaconda/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/BS/sun_project/work/software/Anaconda/lib/pkgconfig:$PKG_CONFIG_PATH

export PATH=:/usr/bin:$PATH
export PATH=:/BS/3d_deep_learning/work/yangTool/mdb/libraries/liblmdb:$PATH
export PATH=:/BS/sun_project/work/software/protobuf/build/bin:$PATH
export PATH=:/BS/sun_project/work/software/gflags-2.0/build/bin:$PATH
export PATH=:/BS/sun_project/work/software/glog-0.3.3/build/lib:$PATH

export LD_LIBRARY_PATH=:/BS/sun_project_meta/work/mlq_project/yyliu_project/packages/cuda-8.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=:/usr/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=:/BS/sun_project/work/software/lmdb/libraries/liblmdb:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=:/BS/sun_project/work/software/protobuf/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=:/BS/sun_project/work/software/snappy-1.1.1/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=:/BS/sun_project/work/software/gflags-2.0/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=:/BS/sun_project/work/software/glog-master/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=:/BS/sun_project/work/software/leveldb:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=:/usr/lib/nvidia-cuda-toolkit/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=:/BS/sun_project/work/software/cudnn-6.0/lib64:$LD_LIBRARY_PATH

INCLUDE_DIR=:/usr/local/include:/usr/include:/usr/include/linux:/BS/sun_project/work/software/cudnn-8.0/include:/BS/sun_project_meta/work/mlq_project/yyliu_project/packages/cuda-8.0/targets/x86_64-linux/include
export CPLUS_INCLUDE="$CPLUS_INCLUDE:$INCLUDE_DIR"
export C_INCLUDE="$C_INCLUDE:$INCLUDE_DIR"

# Activate tensorflow
#source /BS/sun_project2/work/mlq_project/tensorflow/bin/activate
#source /BS/sun_project2/work/mlq_project/venv/bin/activate
source activate tensorflow_1.3.0_gpu
