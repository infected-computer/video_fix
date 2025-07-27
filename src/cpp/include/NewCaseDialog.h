#ifndef NEWCASEDIALOG_H
#define NEWCASEDIALOG_H

#include <QDialog>

namespace Ui {
class NewCaseDialog;
}

class NewCaseDialog : public QDialog
{
    Q_OBJECT

public:
    explicit NewCaseDialog(QWidget *parent = nullptr);
    ~NewCaseDialog();

    QString caseName() const;
    QString examiner() const;
    QString rootDirectory() const;

private slots:
    void on_browseButton_clicked();

private:
    Ui::NewCaseDialog *ui;
};

#endif // NEWCASEDIALOG_H
