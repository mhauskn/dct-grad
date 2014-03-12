import tempfile

class condorJob:
    """
    The Condor Job class is responsible for creating and submitting condor jobs.
    """

    def __init__(self, executable):
        self.group = '+Group = \"GRAD\"'
        self.project = '+Project = \"AI_ROBOTICS\"'
        self.getenv = True
        self.requirements = 'Requirements = Arch == \"X86_64\"'
        self.executable = 'Executable = ' + executable
        self.arguments = ''
        self.output = ''
        self.error = ''
        self.log = ''
        self.nJobs = 1

    def setArgs(self, args):
        self.arguments = 'Arguments = ' + args
        
    def setErrFile(self, error):
        self.error = 'Error = ' + error

    def setOutputFile(self, out):
        self.out = 'Out = ' + out
        
    def setLogFile(self, log):
        self.log = 'Log = ' + log

    def setRequirements(self, requirments):
        self.requirements = 'Requirements = ' + requirements

    def generate(self, fname):
        f = open(fname, 'w')
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
        f.close()

# cj = condorJob('python')
# cj.generate('test.condor')
f = tempfile.NamedTemporaryFile()
print f.name
f.close()

