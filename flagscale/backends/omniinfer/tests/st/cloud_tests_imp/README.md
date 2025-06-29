一. 安装指导
需要按照pytest相关依赖：
pip install pytest requests pytest-html pytest-xdist openpyxl pytest-repeat

二.用例说明
当前用例通过excel数据驱动方式生成用例，用例按照执行时间分为快场景用例和慢场景用，对用execl文件的两个sheet。配置文件在tools目录的1.xlsx
1. 用例执行的判断方式分为： 
--返回码判断， 
--返回值判断，返回值判断分为包含，不包含，以及正则匹配三种方式。
2. 快用例和慢用例通过case_time控制，分别在excel的两个sheet中，指的是执行时长快慢的用例而已，这2份用例需要单独配置2个任务执行，防止时长互相影响。
3. 通过表格的模型列说明该用例适应的模型，目前分为deepseek-r1, 'deepseek-v3，通过配置参数case_type来配置

三. 测试执行
执行命令： 入口函数main.py脚本，执行命令j举例
export PYTHONPATH=$PYTHONPATH:/home/ma-user/ws/pytest_model_test/tools
python main.py --url=http://7.216.55.213:9000/v1/chat/completions --model_name=DeepSeek-v3 --case_type=deepseek-r1 --case_time fast
python main.py --url=http://7.242.107.61:9000/v1/chat/completions --model_name=DeepSeek-v3 --case_type=deepseek-v3 --case_time fast
当前支持的参数如下：（必须配置的是url，model_name, case_type,case_time,model_name)
parser.add_argument('--url', type=str, required=True, help="服务请求全路径")
parser.add_argument('--model_name', type=str, required=True, help="模型名称")
parser.add_argument('--max_fail', '-m', type=int, default='3',
                    help='多少用例执行失败任务停止，服务端异常的话任务能快速停止')
parser.add_argument('--parallel_num', '-n', type=int, default='25',
                    help='并发执行用例')
parser.add_argument('--case_type', '-t', type=str, choices=['deepseek-r1', 'deepseek-v3'],
                    help='用例类型')
parser.add_argument('--case_time', type=str, choices=['fast', 'slow'], default='fast',
                    help='用例类型')


