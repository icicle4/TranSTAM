
ROOTDIR=$1

cd ${ROOTDIR} || exit

mkdir "Split1"
cp *02*.pb Split1
cp *10*.pb Split1
cp *13*.pb Split1

mkdir "Split2"
cp *04*.pb Split2
cp *11*.pb Split2

mkdir "Split3"
cp *05*.pb Split3
cp *09*.pb Split3

mkdir "Split12"
mkdir "Split13"
mkdir "Split23"

cp -r Split1/* Split12/
cp -r Split2/* Split12/
cp -r Split2/* Split23/
cp -r Split3/* Split23/
cp -r Split3/* Split13/
cp -r Split1/* Split13/

python datasets/pre_load_static_dataset.py --dataset_dir ${ROOTDIR}/Split12 --hdf5_path ${ROOTDIR}/train_Split12.hdf5

python datasets/pre_load_static_dataset.py --dataset_dir ${ROOTDIR}/Split13 --hdf5_path ${ROOTDIR}/train_Split13.hdf5

python datasets/pre_load_static_dataset.py --dataset_dir ${ROOTDIR}/Split23 --hdf5_path ${ROOTDIR}/train_Split23.hdf5

python datasets/pre_load_static_dataset.py --dataset_dir ${ROOTDIR}/Split1 --hdf5_path ${ROOTDIR}/eval_Split1.hdf5

python datasets/pre_load_static_dataset.py --dataset_dir ${ROOTDIR}/Split2 --hdf5_path ${ROOTDIR}/eval_Split2.hdf5

python datasets/pre_load_static_dataset.py --dataset_dir ${ROOTDIR}/Split3 --hdf5_path ${ROOTDIR}/eval_Split3.hdf5
