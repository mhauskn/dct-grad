# Regularization Experiment: Investigate how DCT transforms perform with and
# without regularization.
import condor, socket, subprocess

if socket.gethostname() == 'drogba':
    cj = condor.job('python')
    
    # DCT with all regularlization
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/reg-dct')
    cj.setArgs('/home/matthew/projects/dct-grad/autoencoder.py --nEpochs 1000 --nStripes 9 --path /home/matthew/projects/dct-grad/ --outputPrefix reg-dct')
    cj.runLocal()

    # DCT with no KL_Div
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/noKL-dct')
    cj.setArgs('/home/matthew/projects/dct-grad/autoencoder.py --nEpochs 1000 --nStripes 9 --path /home/matthew/projects/dct-grad/ --outputPrefix noKL-dct --noKLDiv')
    cj.runLocal()

    # DCT with no Weight Penalty
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/noWt-dct')
    cj.setArgs('/home/matthew/projects/dct-grad/autoencoder.py --nEpochs 1000 --nStripes 9 --path /home/matthew/projects/dct-grad/ --outputPrefix noWt-dct --noWeightCost')
    cj.runLocal()

    # DCT with only SSE
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/sse-dct')
    cj.setArgs('/home/matthew/projects/dct-grad/autoencoder.py --nEpochs 1000 --nStripes 9 --path /home/matthew/projects/dct-grad/ --outputPrefix sse-dct --noKLDiv --noWeightCost')
    cj.runLocal()

    # noDCT with all regularlization
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/reg')
    cj.setArgs('/home/matthew/projects/dct-grad/autoencoder.py --nEpochs 1000 --nStripes 0 --path /home/matthew/projects/dct-grad/ --outputPrefix reg')
    cj.runLocal()

    # noDCT with no KL_Div
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/noKL')
    cj.setArgs('/home/matthew/projects/dct-grad/autoencoder.py --nEpochs 1000 --nStripes 0 --path /home/matthew/projects/dct-grad/ --outputPrefix noKL --noKLDiv')
    cj.runLocal()

    # noDCT with no Weight Penalty
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/noWt')
    cj.setArgs('/home/matthew/projects/dct-grad/autoencoder.py --nEpochs 1000 --nStripes 0 --path /home/matthew/projects/dct-grad/ --outputPrefix noWt --noWeightCost')
    cj.runLocal()

    # noDCT with only SSE
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/sse')
    cj.setArgs('/home/matthew/projects/dct-grad/autoencoder.py --nEpochs 1000 --nStripes 0 --path /home/matthew/projects/dct-grad/ --outputPrefix sse --noKLDiv --noWeightCost')
    cj.runLocal()
