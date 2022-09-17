'''
Ting BAO @ Tsinghua University
bao-ting@foxmail.com
2022.09.08

this is aiming to scan the inner window and get a good wannier band structure
usage:
prepare several files:
    1. a config file gives out the inner step range, see scan_config.ini
    2. a template of wannier90.win, with 
        dis_froz_min = CONTENT1
        dis_froz_max = CONTENT2
    3. a template run.sh of wannier90.x
    4. data file from VASP projection: wannier90.{amn,eig,mmn}

    (optional,for band plot and visualization)
    5. wannier plot code from fanbs, ask for permission first
    6. BAND.dat, KLABELS
'''

import configparser
import numpy as np
import argparse
import shutil
import os

parser=argparse.ArgumentParser()
parser.add_argument('--config',default='./config.ini',type=str,help='specify the configfile')
args=parser.parse_args()

def from_template(template,content=[]):
    '''A general function, 'CONTENTI' will be replaced by contenti
    content should be a list
    '''
    with open(template) as f:
        temp=f.readlines()
    for i in range(len(temp)):
        for j in range(len(content)):
            temp[i]=temp[i].replace('CONTENT'+str(j+1), str(content[j]))
    return temp


def prepare_file(topath,i,j,template):
    win=from_template(template+'wannier90.win',content=[i,j])
    with open(topath+'wannier90.win','w',encoding='utf-8') as f:
        f.writelines(win)
    for file in ['wannier90.amn','wannier90.mmn','wannier90.eig','run-wannier90.sh','BAND.dat','KLABELS','fitband_fbs.py']:
        shutil.copy(template+file,topath)


def run_and_collect(pathlist,topath,E_fermi):
    runall=[]
    for path in pathlist:
        temp='cd {}&& qsub run-wannier90.sh\n'.format(path)
        runall.append(temp)
    with open(os.path.join(topath,'runall.sh'),'w',encoding='utf-8') as f:
        f.writelines(runall)
    
    # attention: this is a multi processing way, make sure of enough CPU cores and RAM
    collect=[]
    for path in pathlist:
        temp='cd {}&& python fitband_fbs.py|echo {} &\n'.format(path,E_fermi)
        collect.append(temp)
    collect.append('wait\n')
    
    # copy picture to a folder
    if not os.path.exists(os.path.join(topath,'all_figs')):
        os.mkdir(topath+'all_figs')
    for path in pathlist:
        temp ="cp {}fitband.jpg {}all_figs/{}.jpg\n".format(path, topath, path.split('/')[-2])
        collect.append(temp)
    with open(os.path.join(topath,'post_process.sh'),'w',encoding='utf-8') as f:
        f.writelines(collect)
    print("scripts generated")


def main():
    config = configparser.ConfigParser()
    config.read(args.config)
    
    min_start=float(config['dis_froz_min']['start'])
    min_end=float(config['dis_froz_min']['end'])
    min_step=float(config['dis_froz_min']['step'])
    max_start=float(config['dis_froz_max']['start'])
    max_end=float(config['dis_froz_max']['end'])
    max_step=float(config['dis_froz_max']['step'])
    E_fermi=float(config['general']['Fermi_Energy'])
    template=config['general']['template_path']
    output=config['general']['output_path']
    print("config getted")
    
    #check the output path
    if not os.path.exists(output):
        os.mkdir(output)
    
    pathlist=[]
    for i in np.arange(min_start,min_end,min_step):
        for j in np.arange(max_start,max_end,max_step):
            # make and check the folder
            path=os.path.join(output,'{:.3f}_{:.3f}/'.format(i,j))
            pathlist.append(path)
            if not os.path.exists(path):
                os.mkdir(path)
            prepare_file(topath=path,i=i,j=j,template=template)
    
    #prepare the one-key run script and post-process script
    run_and_collect(pathlist,topath=output,E_fermi=E_fermi)

            
if __name__=='__main__':
    main()