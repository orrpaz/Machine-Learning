import torch
from torch import optim, nn

from CNN import MyCNN
from gcommand_loader import GCommandLoader

# Hyper-parameters
# BATCH_SIZE = 40
# IMAGESIZE = 28 * 28
ETA = 0.001
EPOCHS = 10
# FIRST_HIDDEN_LAYER_SIZE = 100
# SECOND_HIDDEN_LAYER_SIZE = 50
# NUMBER_OF_CLASSES = 10
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

class ModelTrainer(object):
    def __init__(self, train_loader, validation_loader, test_loader, model, optimizer):
        """
        initializes the ModelTrainer.
        :param train_loader: training set
        :param validation_loader: validation set
        :param test_loader: test set
        :param model: neural network model
        :param optimizer: optimizer
        """
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def run(self):
        for e in range(EPOCHS+1):
            trainloss=self.train(e)
            valloss = self.valid(e)
        all_pred = self.test()
        return all_pred



    def train(self,e):
        self.model.train()
        total_step = len(self.train_loader)
        total_loss = 0
        loss_list = []
        acc_list = []
        correct = 0
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            if cuda:
                data,labels = data.cuda(),labels.cuda()
            output = self.model(data)
            loss = self.criterion(output,labels)
            loss_list.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total = labels.size(0)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(e + 1, EPOCHS, batch_idx + 1, total_step, loss.item(),
                              (correct / total) * 100))

        return total_loss/len(self.train_loader.dataset)

    def valid(self,e):
        loss_list = []
        total_loss = 0

        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_idx, (data, labels) in enumerate(self.validation_loader):
                if cuda:
                    data,labels = data.cuda(),labels.cuda()
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = self.criterion(output, labels)
                loss_list.append(loss.item())
                total_loss += loss.item()
            total_loss = total_loss / len(self.validation_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                total_loss, correct, len(self.validation_loader.dataset), 100. * correct / len(self.validation_loader.dataset)))
            return total_loss
    def test(self):
        self.model.eval()
        all_pred = []
        for data, target in self.test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            all_pred.extend(pred)
        return all_pred



def write_test(audio_names,all_pred):
    name_list = []
    for name in audio_names:
        name_list.append(name[0].split('/')[len(name[0].split('/')) - 1])
    file = open("test_y", "w")
    for name, pred in zip(name_list,all_pred):
        file.write(name + ', ' + str(pred.item()) + '\n')
    file.close()

def main():
    trainset = GCommandLoader('./sample/train')
    testset = GCommandLoader('./sample/test')
    validationset = GCommandLoader('./sample/valid')

    data_loader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    valid_loader = torch.utils.data.DataLoader(
        validationset, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)
    model = MyCNN(image_size=161*101)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=ETA)
    trainer = ModelTrainer(data_loader, valid_loader, test_loader, model, optimizer)
    all_pred = trainer.run()
    write_test(testset.spects,all_pred)


main()