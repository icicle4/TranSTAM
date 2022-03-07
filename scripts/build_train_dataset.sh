ROOTDIR=$1

cd ${ROOTDIR} || exit


mkdir "train"

cp *02*.pb "train"
cp *10*.pb "train"
cp *13*.pb "train"
cp *04*.pb "train"
cp *11*.pb "train"
cp *05*.pb "train"
cp *09*.pb "train"

mkdir "test"


cp *01*.pb "test"
cp *03*.pb "test"
cp *06*.pb "test"
cp *07*.pb "test"
cp *08*.pb "test"
cp *12*.pb "test"
cp *14*.pb "test"

