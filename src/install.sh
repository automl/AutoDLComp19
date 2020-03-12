# swig dependency
tar -xvzf swig-4.0.1.tar.gz
cd swig-4.0.1

#./configure --prefix=/home/yourname/projects
./configure
make
make install

# AutoFolio requirements
pip install -r af_requirements.txt
