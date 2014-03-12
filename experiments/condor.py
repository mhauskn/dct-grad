import tempfile, subprocess

class job:
    def __init__(self, executable):
        self.group        = '+Group = \"GRAD\"'
        self.project      = '+Project = \"AI_ROBOTICS\"'
        self.getenv       = True
        self.requirements = 'Requirements = Arch == \"X86_64\"'
        self.executable   = 'Executable = ' + executable
        self.arguments    = ''
        self.output       = ''
        self.error        = ''
        self.log          = ''
        self.nJobs        = 1

    def setArgs(self, args):
        self.arguments = 'Arguments = ' + args
        
    def setOutputPrefix(self, prefix):
        self.setErr(prefix + '.err')
        self.setOutput(prefix + '.out')
        self.setLog(prefix + '.log')

    def setErr(self, error):
        self.error = 'Error = ' + error

    def setOutput(self, out):
        self.output = 'Output = ' + out
        
    def setLog(self, log):
        self.log = 'Log = ' + log

    def setRequirements(self, requirments):
        self.requirements = 'Requirements = ' + requirements

    def submit(self):
        f = tempfile.NamedTemporaryFile()
        f.write(self.group+'\n')
        f.write(self.project+'\n')
        if self.getenv:
            f.write('getenv = true'+'\n')
        f.write(self.executable+'\n')
        f.write(self.arguments+'\n')
        f.write(self.requirements+'\n')
        f.write(self.error+'\n')
        f.write(self.output+'\n')
        f.write(self.log+'\n')
        f.write('Queue ' + str(self.nJobs) +'\n')
        f.flush()
        condorFile = f.name
        output = subprocess.Popen(["condor_submit","-verbose",condorFile],stdout=subprocess.PIPE).communicate()[0]   
        f.close()
        s = output.find('** Proc ')+8
        procID = output[s:output.find(':\n',s)]
        return procID
