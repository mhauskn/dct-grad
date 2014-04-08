import tempfile, subprocess

class Job:
    def __init__(self, executable, args=''):
        self.executable = executable
        self.output = 'job.out'
        self.error = 'job.err'
        self.arguments = args

    def setArgs(self, args):
        self.arguments = args
        
    def setOutputPrefix(self, prefix):
        self.setErr(prefix + '.err')
        self.setOutput(prefix + '.out')
        self.setLog(prefix + '.log')

    def setErr(self, error):
        self.error = error

    def setOutput(self, out):
        self.output = out
        
    def setLog(self, log):
        self.log = log

    def submit(self):
        ofile      = self.output
        efile      = self.error
        executable = self.executable
        arguments  = self.arguments
        z = arguments.split(' ')
        z.insert(0,executable)

        out = open(ofile,'w')
        err = open(efile,'w')
        subprocess.Popen(z,stdout=out,stderr=err).wait()
        out.close()
        err.close()


class CondorJob(Job):
    def __init__(self, executable, args=''):
        Job.__init__(self, executable, args)

    def submit(self):
        f = tempfile.NamedTemporaryFile()
        f.write('+Group = \"GRAD\"\n')
        f.write('+Project = \"AI_ROBOTICS\"\n')
        f.write('getenv = true\n')
        f.write('Executable = '+self.executable+'\n')
        f.write('Arguments = '+self.arguments+'\n')
        f.write('Requirements = Arch == \"X86_64\"\n')
        f.write('Error = '+self.error+'\n')
        f.write('Output = '+self.output+'\n')
        f.write('Log = '+self.log+'\n')
        f.write('Queue\n')
        f.flush()
        condorFile = f.name
        output = subprocess.Popen(["condor_submit","-verbose",condorFile],stdout=subprocess.PIPE).communicate()[0]   
        f.close()
        s = output.find('** Proc ')+8
        procID = output[s:output.find(':\n',s)]
        return procID

class TaccJob(Job):
    def __init__(self, executable, args=''):
        Job.__init__(self, executable, args)
        self.hours = 0
        self.minutes = 30

    def setJobTime(self, hours, minutes):
        assert(minutes < 60 and minutes >= 0)
        assert(hours >= 0)
        self.hours = hours
        self.minutes = minutes

    def submit(self):
        f = tempfile.NamedTemporaryFile()
        f.write('#!/bin/bash\n')
        f.write('#SBATCH -J dct-grad\n')
        f.write('#SBATCH -o '+self.output+'\n')
        f.write('#SBATCH -e '+self.error+'\n')        
        f.write('#SBATCH -p gpu\n')
        f.write('#SBATCH -N 1\n')
        f.write('#SBATCH -n 20\n')
        f.write('#SBATCH -t '+str(self.hours)+':'+str(self.minutes)+':00\n')        
        f.write(self.executable+' '+self.arguments+'\n')
        f.flush()
        jobFile = f.name
        subprocess.Popen(["sbatch",jobFile],stdout=subprocess.PIPE).communicate()[0]   
        f.close()
        
