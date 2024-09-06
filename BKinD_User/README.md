# BKinD readme.txt with Setup Instructions

Welcome to BKinD! This guide will walk you through the steps necessary to set up the required environment and install all necessary software to use BKinD effectively. BKinD (BKinD Kernel in Diffraction) is a tool designed for filtering and analyzing Diffraction Data. Please follow the instructions to ensure a successful setup.

This guide is divided into the following sections:

System Requirements

The following must be manually installed by the user before running the script. If these are not installed, the script will terminate and display an error message.

   1. Install Conda (see instructions below)
   2. Install SHELXL (see instructions below)
   3. Set Up XDS (see instructions below)

Once these are installed, you can run the BKinD application in two ways:

Via Command Line:
Navigate to the BKinD folder:

cd path/to/BKinD

Ensure that your Conda environment is activated and both SHELXL and XDS are runnable. Then run the application:

python bkind.py

For Python 3.x versions:

python3 bkind.py

Via run_script.bat:
You can also run the application by double-clicking the pre-configured batch script run_script.bat, which will activate the Conda environment and run the BKinD application.

To run it:

Double-click run_script.bat in the BKinD folder.
Creating a Shortcut for run_script.bat:
To make launching BKinD easier, you can create a desktop shortcut for run_script.bat and customize it with an icon from the assets folder:

Right-click run_script.bat and select Create Shortcut.
Right-click the newly created shortcut and select Properties.
In the Shortcut tab, click Change Icon....
Browse to the assets folder and select an icon (e.g., assets/icon.ico).
Click OK and Apply to set the new icon.
Now you can double-click the shortcut on your desktop to run BKinD.


The script will then update python to necessary version and install/update necessary packages.

This can also be done manually using the following instructions further down in this document:

   4. Instructions for Setting Up Your Conda Environment Manually using requirements.txt

There is also an explanation of the script output:
 
   5. Explanation of Output

If you encounter any issues during the installation process, please refer to the respective official documentation or seek assistance from the BKinD support team (buster.blomberg@mmk.su.se).


#####################################################################
############ Instructions for Installing and Using Conda ############
#####################################################################

## Installing Conda

1. **Download the Conda Installer**:

   - Visit the official Conda download page: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
   - Choose the appropriate installer (Anaconda or Miniconda) for your operating system (Windows, Mac, Linux).

2. **Run the Installer**:

   - For **Windows**:
     1. Download the `.exe` installer.
     2. Double-click the installer to launch it.
     3. Follow the on-screen instructions to complete the installation.
   
   - For **Mac**:

     1. Download the `.sh` installer.
     2. Open a terminal and run the following command to execute the installer:

        bash Miniconda3-latest-MacOSX-x86_64.sh

     3. Follow the on-screen instructions to complete the installation.

   - For **Linux**:

     1. Download the `.sh` installer.
     2. Open a terminal and run the following command to execute the installer:

        bash Miniconda3-latest-Linux-x86_64.sh

     3. Follow the on-screen instructions to complete the installation.

3. **Restart Your Terminal**:

   - Close and reopen your terminal to apply the changes.

4. **Verify the Installation**:

   - To ensure Conda is installed correctly, run the following command:

     conda --version

   - You should see the Conda version information displayed.

## Using Conda

1. **Create a New Conda Environment** (optional but recommended):

   - Open your terminal and run the following command to create a new Conda environment named `bkind-env` (you can choose a different name if you prefer):

     conda create -n bkind-env


2. **Activate the Conda Environment**:

   - Once the environment is created, activate it with the following command:

     conda activate bkind-env


3. **Install Packages**:

   - To install packages in your environment, use the following command:

     conda install package_name

   - Replace `package_name` with the name of the package you want to install. You can install multiple packages at once:

     conda install numpy pandas


4. **List Installed Packages**:

   - To see a list of all packages installed in your environment, use:

     conda list


5. **Update Packages**:
   - To update a specific package to the latest version, use:

     conda update package_name


6. **Remove Packages**:
   - To remove a specific package from your environment, use:

     conda remove package_name


7. **Deactivate the Conda Environment**:
   - To deactivate your current environment and return to the base environment, use:

     conda deactivate


8. **Delete a Conda Environment**:
   - To delete a Conda environment, use:

     conda env remove -n environment_name

   - Replace `environment_name` with the name of the environment you want to delete.

By following these instructions, you will successfully install Conda on your system and be able to manage your Conda environments and packages effectively.

########################################################
############ Instructions to Install SHELXL ############
########################################################

1. **Register as an Academic User**:

   - Visit the website: http://shelx.uni-goettingen.de/register.html
   - Complete the registration form. The answer to the "Xtal" question is P212121.

2. **Download the SHELXL Software**:

   - Visit the website: http://shelx.uni-goettingen.de/
   - Download the appropriate version for your system (Windows, Mac, Linux) from: https://shelx.uni-goettingen.de/bin/

3. **Extract the Downloaded Package**:

   - Extract the contents of the downloaded package.

4. **Move the SHELXL Executable to a Directory Included in Your System's PATH**:

   - For **Linux and Mac**:

     sudo mv shelxl /usr/local/bin

   - For **Windows**:

     - Move the extracted `shelxl.exe` file to a directory included in your PATH, such as `C:\Windows\System32`.

5. **Verify the Installation**:
   - Run the following command(in the terminal) to verify the installation:

     shelxl

   - If the installation is successful, you should see the SHELXL version information.

#########################################################
############ Instructions for Setting Up XDS ############
#########################################################

## Instructions for Setting Up XDS on Linux

1. **Become root** (on Ubuntu, use `sudo -i`, it will ask for password):
   sudo -i

2. **Change directory to `/usr/local/bin`**:
   cd /usr/local/bin

3. **Download and extract the XDS software**:
   wget -O- https://xds.mr.mpg.de/XDS-INTEL64_Linux_x86_64.tar.gz | tar xzvf -

4. **Create symbolic links for the extracted files**:
   ln -sf XDS-INTEL64_Linux_x86_64/* .

## Instructions for Setting Up XDS on Mac with Intel/Apple M Processors

1. **Obtain root privileges** (it will ask for your password):
   sudo su

2. **Change directory to `/usr/local/bin`**:
   cd /usr/local/bin

3. **Download and extract the XDS software**:
   curl -L -o - https://xds.mr.mpg.de/XDS-OSX_64.tar.gz | tar xzvf -

4. **Create symbolic links for the extracted files**:
   - For Intel:
     ln -sf XDS-OSX_64/* .
   - For Apple M:
     ln -sf XDS-Apple_M1/* .

5. **Release root permissions**:
   exit

By following these instructions, you will successfully set up XDS on your system. If you encounter any issues or need further assistance, please refer to the official XDS documentation or reach out for help. See https://wiki.uni-konstanz.de/xds/index.php/Installation from where these instructions are taken.

#####################################################################################
############ Instructions for Setting Up Your Conda Environment MANUALLY ############
#####################################################################################

## Instructions for Setting Up Your Conda Environment

1. **Create a New Conda Environment** (optional but recommended):
   Open your terminal and run the following command to create a new Conda environment with required python version(3.12.2) named `bkind-env` (you can choose a different name if you prefer):
   conda create -n bkind-env python=3.12.2 -y

2. **Activate the Conda Environment**:
   Once the environment is created, activate it with the following command:
   conda activate bkind-env

3. **Install `cctbx-base` from `conda-forge`**:
   Finally, install `cctbx-base` from the `conda-forge` channel by running:
   conda install -c conda-forge cctbx-base -y

4. **Change Directory to the BKinD Folder**:
   Navigate to the BKinD setup folder where your `requirements.txt` file is located. Use the `cd` command to change the directory:
   cd path/to/BKinD/setup

5. **Install Packages from `requirements.txt`**:
   Run the following command to install all specified packages from the `requirements.txt` file:
   conda install --file requirements.txt -y

6. **Install `tkinter`** (for Ubuntu users with limited installations):
   If you are using Ubuntu, you might need to install `tkinter` separately using the following command:
   sudo apt-get install python3-tk

7. **Run BKinD**
   You're now set to run the application using
	python bkind.py

By following these instructions, you will set up your Conda environment with all the required packages specified in your `requirements.txt` file. If you have any issues or questions, please feel free to reach out for further assistance.

#######################################################################
############ Output Folder Contents and Their Descriptions ############
#######################################################################

 	filtering_stats.txt:

This text file contains statistics related to the filtering process. The file contains the following types of data:

Original and Remaining Data Counts: Details the initial number of reflections and unique ASU, as well as the remaining counts after filtering.
Filtering Parameters: Includes the target ASU percentage and the number of iterations required to reach that target.
Data Quality Metrics: Provides various metrics such as R1, FVAR, Rint, highest diffraction peak, deepest hole, and one sigma level, which assess the quality and accuracy of the filtered data.
Filtering Effectiveness: Shows the percentage of data remaining after filtering, indicating how much data was retained or removed at each filtering stage.
	
 	aggregated_filtered:

This folder contains aggregated data .csv files from the filtered datasets. It includes combined or summarized results from the individual filtered files.

	filtered_90.0 to filtered_100.0:

The files in these folders represent filtered datasets at various threshold levels (90.0, 91.0, 92.0, ..., 100.0). The numbers correspond to the percentage of data retained after filtering. These are aggregated in the folder aggregated_filtered.

	INTEGRATE.HKL:

This file contains integrated reflection intensities. It is generated during the data processing of X-ray diffraction experiments and includes information about the reflections, such as their indices, intensities, and other relevant parameters.

	XDSCONV.LP:

This log file is produced by the XDSCONV program, which is part of the XDS suite for processing X-ray diffraction data. It contains information about the conversion of data formats, such A from XDS to other formats used in crystallography.

	XDS_ASCII_NEM.HKL:

This file is another type of reflection data file, in ASCII format, which can be used for various downstream analysis steps in crystallography.

	bkind.fcf:

This file format (.fcf) contains final structure factor amplitudes and phases, which are crucial for determining the electron density map in crystallography.

	bkind.hkl:

This file contains reflection data to be refined using SHELXL.

	bkind.ins:

This file is an instruction file for the refinement program SHELXL, which is used to refine the crystal structure model. It contains initial parameters and instructions for the refinement process.

	bkind.lst:

This list file contains detailed output from the refinement process, including diagnostics, errors, and final refined parameters. It is generated by SHELXL.

	bkind.mat:

This file contain matrix data, related to the orientation or transformation matrices used in crystallography data processing.

	bkind.res:

This file is a result file from the refinement process, containing the refined atomic coordinates and other relevant structural information.

	xdsconv.inp:

This input file is used by the XDSCONV program. It contains parameters and instructions for converting XDS output files to other formats used in crystallographic analysis.

