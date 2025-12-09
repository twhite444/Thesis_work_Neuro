@echo off
echo Activating conda environment...
call C:\ProgramData\Anaconda3\Scripts\activate.bat PyrfumeThesis

echo Installing required packages...
pip install torch torch-geometric pandas numpy scikit-learn matplotlib seaborn rdkit networkx

echo Installation complete!
pause 