#ifndef WORKFLOW_UI_H
#define WORKFLOW_UI_H

#include <QWidget>
#include <QDialog>
#include <QWizard>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QTimer>
#include <QPropertyAnimation>
#include <QGraphicsEffect>
#include <memory>

// Forward declarations
class WorkflowEngine;
class WorkflowNode;
class WorkflowConnection;
class WorkflowTask;

/**
 * @brief Main workflow management interface
 */
class WorkflowManager : public QWidget {
    Q_OBJECT

public:
    explicit WorkflowManager(QWidget* parent = nullptr);
    ~WorkflowManager();

    // Workflow management
    void createNewWorkflow();
    void openWorkflow(const QString& filePath);
    void saveWorkflow(const QString& filePath = QString());
    void closeWorkflow();
    
    // Workflow execution
    void executeWorkflow();
    void pauseWorkflow();
    void stopWorkflow();
    void resumeWorkflow();
    
    // View management
    void setViewMode(const QString& mode);
    QString getViewMode() const;
    QStringList getAvailableViewModes() const;
    
    // Current workflow access
    class WorkflowEditor* getWorkflowEditor() const;
    class WorkflowMonitor* getWorkflowMonitor() const;
    class WorkflowTemplates* getWorkflowTemplates() const;
    
signals:
    void workflowCreated(const QString& workflowId);
    void workflowOpened(const QString& workflowId);
    void workflowSaved(const QString& workflowId);
    void workflowClosed(const QString& workflowId);
    void workflowExecutionStarted(const QString& workflowId);
    void workflowExecutionFinished(const QString& workflowId, bool success);
    
private slots:
    void onNewWorkflow();
    void onOpenWorkflow();
    void onSaveWorkflow();
    void onSaveWorkflowAs();
    void onCloseWorkflow();
    void onExecuteWorkflow();
    void onStopWorkflow();
    void onViewModeChanged();
    void onWorkflowChanged();
    
private:
    class QTabWidget* m_tabWidget;
    class QToolBar* m_toolBar;
    class QSplitter* m_splitter;
    
    class WorkflowEditor* m_workflowEditor;
    class WorkflowMonitor* m_workflowMonitor;
    class WorkflowTemplates* m_workflowTemplates;
    
    QString m_currentWorkflowId;
    QString m_viewMode;
    QStringList m_viewModes;
    
    void setupUI();
    void createToolBar();
    void createWorkflowEditor();
    void createWorkflowMonitor();
    void createWorkflowTemplates();
    void connectSignals();
    void updateUI();
};

/**
 * @brief Visual workflow editor with drag-and-drop interface
 */
class WorkflowEditor : public QWidget {
    Q_OBJECT

public:
    explicit WorkflowEditor(QWidget* parent = nullptr);
    ~WorkflowEditor();

    // Workflow operations
    void newWorkflow();
    void loadWorkflow(const QString& data);
    QString saveWorkflow() const;
    void clearWorkflow();
    
    // Node management
    void addNode(const QString& nodeType, const QPointF& position);
    void removeNode(const QString& nodeId);
    void selectNode(const QString& nodeId);
    QStringList getSelectedNodes() const;
    
    // Connection management
    void connectNodes(const QString& fromNodeId, const QString& toNodeId);
    void disconnectNodes(const QString& fromNodeId, const QString& toNodeId);
    
    // View operations
    void zoomIn();
    void zoomOut();
    void zoomToFit();
    void resetZoom();
    void centerView();
    
    // Grid and alignment
    void setGridEnabled(bool enabled);
    bool isGridEnabled() const;
    void setSnapToGrid(bool enabled);
    bool isSnapToGrid() const;
    void alignSelectedNodes(const QString& alignment);
    
    // Workflow validation
    bool validateWorkflow() const;
    QStringList getValidationErrors() const;
    
signals:
    void workflowChanged();
    void nodeSelected(const QString& nodeId);
    void nodeDoubleClicked(const QString& nodeId);
    void nodesConnected(const QString& fromNodeId, const QString& toNodeId);
    void nodesDisconnected(const QString& fromNodeId, const QString& toNodeId);
    
private slots:
    void onSceneChanged();
    void onSelectionChanged();
    void onNodeDoubleClicked();
    void onContextMenuRequested(const QPoint& position);
    void onDeleteSelectedNodes();
    void onCopySelectedNodes();
    void onPasteNodes();
    void onUndoAction();
    void onRedoAction();
    
private:
    class WorkflowScene* m_scene;
    class WorkflowView* m_view;
    class QToolBar* m_toolBar;
    class NodePalette* m_nodePalette;
    
    bool m_gridEnabled;
    bool m_snapToGrid;
    int m_gridSize;
    
    QMap<QString, WorkflowNode*> m_nodes;
    QList<WorkflowConnection*> m_connections;
    
    // Undo/Redo system
    class QUndoStack* m_undoStack;
    
    void setupUI();
    void createScene();
    void createView();
    void createToolBar();
    void createNodePalette();
    void setupUndoRedo();
    void drawGrid();
    QPointF snapToGrid(const QPointF& position) const;
};

/**
 * @brief Custom graphics scene for workflow editing
 */
class WorkflowScene : public QGraphicsScene {
    Q_OBJECT

public:
    explicit WorkflowScene(QObject* parent = nullptr);
    ~WorkflowScene();

    // Node management
    WorkflowNode* addNode(const QString& nodeType, const QPointF& position);
    void removeNode(WorkflowNode* node);
    WorkflowNode* getNode(const QString& nodeId) const;
    QList<WorkflowNode*> getAllNodes() const;
    
    // Connection management
    WorkflowConnection* addConnection(WorkflowNode* fromNode, WorkflowNode* toNode);
    void removeConnection(WorkflowConnection* connection);
    QList<WorkflowConnection*> getConnections(WorkflowNode* node) const;
    
    // Grid properties
    void setGridSize(int size);
    int getGridSize() const;
    void setGridVisible(bool visible);
    bool isGridVisible() const;
    
protected:
    void drawBackground(QPainter* painter, const QRectF& rect) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void dragEnterEvent(QGraphicsSceneDragDropEvent* event) override;
    void dragMoveEvent(QGraphicsSceneDragDropEvent* event) override;
    void dropEvent(QGraphicsSceneDragDropEvent* event) override;
    
signals:
    void nodeAdded(WorkflowNode* node);
    void nodeRemoved(WorkflowNode* node);
    void connectionAdded(WorkflowConnection* connection);
    void connectionRemoved(WorkflowConnection* connection);
    void nodePositionChanged(WorkflowNode* node);
    
private slots:
    void onNodePositionChanged();
    
private:
    int m_gridSize;
    bool m_gridVisible;
    bool m_connecting;
    WorkflowNode* m_connectionStart;
    QGraphicsLineItem* m_connectionLine;
    
    QMap<QString, WorkflowNode*> m_nodes;
    QList<WorkflowConnection*> m_connections;
    
    void drawGrid(QPainter* painter, const QRectF& rect);
    void startConnection(WorkflowNode* fromNode, const QPointF& position);
    void updateConnection(const QPointF& position);
    void finishConnection(WorkflowNode* toNode);
    void cancelConnection();
};

/**
 * @brief Custom graphics view for workflow editing
 */
class WorkflowView : public QGraphicsView {
    Q_OBJECT

public:
    explicit WorkflowView(QWidget* parent = nullptr);
    explicit WorkflowView(QGraphicsScene* scene, QWidget* parent = nullptr);
    ~WorkflowView();

    // Zoom operations
    void zoomIn();
    void zoomOut();
    void zoomToFit();
    void resetZoom();
    void setZoomFactor(double factor);
    double getZoomFactor() const;
    
    // View operations
    void centerOn(const QPointF& position);
    void centerOnSelection();
    
protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    
signals:
    void zoomChanged(double factor);
    void viewChanged();
    
private:
    double m_zoomFactor;
    double m_minZoom;
    double m_maxZoom;
    bool m_panning;
    QPoint m_lastPanPos;
    
    void initializeView();
    void updateZoom(double factor);
};

/**
 * @brief Workflow node representing a task or operation
 */
class WorkflowNode : public QGraphicsItem {
public:
    enum NodeType {
        InputNode,
        ProcessingNode,
        OutputNode,
        DecisionNode,
        LoopNode,
        CustomNode
    };

    explicit WorkflowNode(const QString& nodeId, NodeType type = ProcessingNode);
    ~WorkflowNode();

    // Node properties
    QString getNodeId() const;
    void setNodeType(NodeType type);
    NodeType getNodeType() const;
    void setNodeTitle(const QString& title);
    QString getNodeTitle() const;
    void setNodeDescription(const QString& description);
    QString getNodeDescription() const;
    
    // Node configuration
    void setNodeParameters(const QVariantMap& parameters);
    QVariantMap getNodeParameters() const;
    void setParameter(const QString& key, const QVariant& value);
    QVariant getParameter(const QString& key) const;
    
    // Visual properties
    void setNodeColor(const QColor& color);
    QColor getNodeColor() const;
    void setNodeIcon(const QIcon& icon);
    QIcon getNodeIcon() const;
    
    // Connection points
    QPointF getInputConnectionPoint() const;
    QPointF getOutputConnectionPoint() const;
    QList<QPointF> getConnectionPoints() const;
    
    // State management
    void setNodeState(const QString& state);
    QString getNodeState() const;
    void setNodeProgress(int progress);
    int getNodeProgress() const;
    
    // QGraphicsItem interface
    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
    QPainterPath shape() const override;
    
protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) override;
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;
    
private:
    QString m_nodeId;
    NodeType m_nodeType;
    QString m_title;
    QString m_description;
    QVariantMap m_parameters;
    QColor m_color;
    QIcon m_icon;
    QString m_state;
    int m_progress;
    
    // Visual properties
    static constexpr double NODE_WIDTH = 120.0;
    static constexpr double NODE_HEIGHT = 80.0;
    static constexpr double CORNER_RADIUS = 8.0;
    
    void updateAppearance();
    QColor getStateColor() const;
    void drawProgressIndicator(QPainter* painter, const QRectF& rect);
};

/**
 * @brief Connection between workflow nodes
 */
class WorkflowConnection : public QGraphicsItem {
public:
    explicit WorkflowConnection(WorkflowNode* fromNode, WorkflowNode* toNode);
    ~WorkflowConnection();

    // Connection properties
    WorkflowNode* getFromNode() const;
    WorkflowNode* getToNode() const;
    void updatePath();
    
    // Visual properties
    void setConnectionColor(const QColor& color);
    QColor getConnectionColor() const;
    void setConnectionWidth(double width);
    double getConnectionWidth() const;
    void setConnectionStyle(Qt::PenStyle style);
    Qt::PenStyle getConnectionStyle() const;
    
    // State
    void setConnectionState(const QString& state);
    QString getConnectionState() const;
    void setDataFlow(bool hasFlow);
    bool hasDataFlow() const;
    
    // QGraphicsItem interface
    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
    QPainterPath shape() const override;
    
protected:
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;
    
private slots:
    void onNodePositionChanged();
    
private:
    WorkflowNode* m_fromNode;
    WorkflowNode* m_toNode;
    QPainterPath m_path;
    QColor m_color;
    double m_width;
    Qt::PenStyle m_style;
    QString m_state;
    bool m_hasDataFlow;
    
    // Animation
    QPropertyAnimation* m_flowAnimation;
    double m_flowPosition;
    
    void calculatePath();
    void setupAnimation();
    void drawArrowHead(QPainter* painter, const QPointF& position, double angle);
    void drawDataFlow(QPainter* painter);
};

/**
 * @brief Node palette for adding nodes to workflow
 */
class NodePalette : public QWidget {
    Q_OBJECT

public:
    explicit NodePalette(QWidget* parent = nullptr);
    ~NodePalette();

    // Category management
    void addCategory(const QString& category);
    void removeCategory(const QString& category);
    QStringList getCategories() const;
    
    // Node types
    void addNodeType(const QString& category, const QString& nodeType, 
                    const QString& title, const QString& description, const QIcon& icon);
    void removeNodeType(const QString& nodeType);
    
    // Search and filter
    void setSearchFilter(const QString& filter);
    QString getSearchFilter() const;
    void setCategoryFilter(const QString& category);
    QString getCategoryFilter() const;
    
signals:
    void nodeTypeSelected(const QString& nodeType);
    void nodeTypeDragStarted(const QString& nodeType);
    
private slots:
    void onCategoryChanged();
    void onSearchTextChanged();
    void onNodeTypeClicked();
    void onNodeTypeDragStarted();
    
private:
    class QTreeWidget* m_nodeTree;
    class QLineEdit* m_searchEdit;
    class QComboBox* m_categoryCombo;
    
    struct NodeTypeInfo {
        QString nodeType;
        QString title;
        QString description;
        QIcon icon;
        QString category;
    };
    
    QMap<QString, NodeTypeInfo> m_nodeTypes;
    QString m_searchFilter;
    QString m_categoryFilter;
    
    void setupUI();
    void populateNodeTree();
    void filterNodes();
    void createBuiltinNodeTypes();
};

/**
 * @brief Workflow execution monitor
 */
class WorkflowMonitor : public QWidget {
    Q_OBJECT

public:
    explicit WorkflowMonitor(QWidget* parent = nullptr);
    ~WorkflowMonitor();

    // Monitoring operations
    void startMonitoring(const QString& workflowId);
    void stopMonitoring();
    void updateNodeStatus(const QString& nodeId, const QString& status, int progress = -1);
    void updateWorkflowStatus(const QString& status);
    
    // Display options
    void setRefreshInterval(int intervalMs);
    int getRefreshInterval() const;
    void setAutoScroll(bool enabled);
    bool isAutoScrollEnabled() const;
    
    // Data export
    void exportExecutionLog(const QString& filePath);
    void clearExecutionLog();
    
signals:
    void executionLogUpdated();
    void nodeStatusChanged(const QString& nodeId, const QString& status);
    void workflowStatusChanged(const QString& status);
    
private slots:
    void onRefreshTimer();
    void onClearLog();
    void onExportLog();
    void onAutoScrollToggled();
    
private:
    class QTreeWidget* m_executionTree;
    class QTextEdit* m_logOutput;
    class QSplitter* m_splitter;
    class QToolBar* m_toolBar;
    class QTimer* m_refreshTimer;
    
    QString m_currentWorkflowId;
    bool m_monitoring;
    bool m_autoScroll;
    int m_refreshInterval;
    
    struct ExecutionEntry {
        QString nodeId;
        QString status;
        int progress;
        QDateTime timestamp;
        QString message;
    };
    
    QList<ExecutionEntry> m_executionLog;
    
    void setupUI();
    void createExecutionTree();
    void createLogOutput();
    void createToolBar();
    void updateExecutionDisplay();
    void addLogEntry(const QString& message, const QString& level = "INFO");
};

/**
 * @brief Workflow templates manager
 */
class WorkflowTemplates : public QWidget {
    Q_OBJECT

public:
    explicit WorkflowTemplates(QWidget* parent = nullptr);
    ~WorkflowTemplates();

    // Template management
    void loadTemplates();
    void saveTemplate(const QString& name, const QString& workflowData, 
                     const QString& description = QString());
    void deleteTemplate(const QString& name);
    void importTemplate(const QString& filePath);
    void exportTemplate(const QString& name, const QString& filePath);
    
    // Template categories
    void addCategory(const QString& category);
    void removeCategory(const QString& category);
    QStringList getCategories() const;
    
    // Template information
    QStringList getTemplateNames() const;
    QString getTemplateData(const QString& name) const;
    QString getTemplateDescription(const QString& name) const;
    QString getTemplateCategory(const QString& name) const;
    
signals:
    void templateSelected(const QString& name, const QString& data);
    void templateDoubleClicked(const QString& name, const QString& data);
    void templatesChanged();
    
private slots:
    void onTemplateSelectionChanged();
    void onTemplateDoubleClicked();
    void onNewTemplate();
    void onDeleteTemplate();
    void onImportTemplate();
    void onExportTemplate();
    void onCategoryChanged();
    
private:
    class QTreeWidget* m_templateTree;
    class QTextEdit* m_descriptionEdit;
    class QComboBox* m_categoryCombo;
    class QToolBar* m_toolBar;
    class QSplitter* m_splitter;
    
    struct TemplateInfo {
        QString name;
        QString data;
        QString description;
        QString category;
        QDateTime created;
        QDateTime modified;
    };
    
    QMap<QString, TemplateInfo> m_templates;
    QString m_templatesPath;
    
    void setupUI();
    void createTemplateTree();
    void createDescriptionPanel();
    void createToolBar();
    void populateTemplateTree();
    void updateTemplateDisplay();
    void loadBuiltinTemplates();
    QString getTemplatesDirectory() const;
};

/**
 * @brief Workflow creation wizard
 */
class WorkflowWizard : public QWizard {
    Q_OBJECT

public:
    explicit WorkflowWizard(QWidget* parent = nullptr);
    ~WorkflowWizard();

    // Wizard results
    QString getWorkflowName() const;
    QString getWorkflowDescription() const;
    QString getWorkflowType() const;
    QVariantMap getWorkflowConfiguration() const;
    QString getGeneratedWorkflow() const;
    
private:
    class WelcomePage* m_welcomePage;
    class WorkflowTypePage* m_typePage;
    class ConfigurationPage* m_configPage;
    class SummaryPage* m_summaryPage;
    
    void createPages();
};

#endif // WORKFLOW_UI_H