#include "include/NewCaseDialog.h"
#include "ui_NewCaseDialog.h"
#include <QFileDialog>
#include <QStandardPaths>

NewCaseDialog::NewCaseDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::NewCaseDialog)
{
    ui->setupUi(this);
    ui->rootDirectoryLineEdit->setText(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
}

NewCaseDialog::~NewCaseDialog()
{
    delete ui;
}

QString NewCaseDialog::caseName() const
{
    return ui->caseNameLineEdit->text();
}

QString NewCaseDialog::examiner() const
{
    return ui->examinerLineEdit->text();
}

QString NewCaseDialog::rootDirectory() const
{
    return ui->rootDirectoryLineEdit->text();
}

void NewCaseDialog::on_browseButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, "Select Root Directory", ui->rootDirectoryLineEdit->text());
    if (!dir.isEmpty()) {
        ui->rootDirectoryLineEdit->setText(dir);
    }
}
